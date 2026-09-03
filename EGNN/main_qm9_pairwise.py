"""Train EGNN and HyEGNN variants on QM9."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from qm9 import dataset
from qm9 import utils as qm9_utils
from qm9.cuda_graphs import (
    BucketedCudaGraphRunner,
    replace_linear_layers_for_graph_capture,
)
from qm9.data.collate import collate_fn as base_collate
from qm9.models_pairwise import (
    EGNN,
    HybridEGNN,
    PairwiseEGNN,
    SparseEdgeCollate,
    SparseEGNN,
    assemble_batch_sparse_edges,
    precompute_molecule_colorings,
)
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader, Subset

EGNN_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = EGNN_ROOT.parent
DEFAULT_OUTPUT_DIRECTORY = EGNN_ROOT / "qm9" / "logs"
PROFILE_BATCH_COUNT = int(os.environ.get("EGNN_PROFILE", "0"))


@dataclass(frozen=True)
class ModelInfo:
    model: nn.Module
    name: str
    hidden_features: int
    pairwise_features: int | None
    sparse_layer_count: int
    uses_sparse_edges: bool


@dataclass
class TrainingContext:
    args: argparse.Namespace
    model_info: ModelInfo
    optimizer: optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    loss_function: nn.Module
    device: torch.device
    charge_scale: Any
    target_mean: Tensor | float
    target_deviation: Tensor | float
    coloring_cache: Any
    graph_runner: BucketedCudaGraphRunner | None


class PhaseProfiler:
    """Collect host wall time and CUDA event time for training phases."""

    def __init__(self, phase_names: list[str]) -> None:
        self.phase_names = phase_names
        self.host_seconds = {name: 0.0 for name in phase_names}
        self.cuda_events = {name: [] for name in phase_names}
        self.batch_count = 0
        self._start_time = 0.0
        self._start_event: torch.cuda.Event | None = None
        self._active_phase = ""

    def start(self, phase_name: str) -> None:
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._start_event.record()
        self._start_time = time.perf_counter()
        self._active_phase = phase_name

    def stop(self) -> None:
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        self.host_seconds[self._active_phase] += time.perf_counter() - self._start_time
        assert self._start_event is not None
        self.cuda_events[self._active_phase].append((self._start_event, end_event))

    def report(self, wall_seconds: float) -> None:
        torch.cuda.synchronize()
        print(f"\n=== phase profile over {self.batch_count} batches (ms/batch) ===")
        print(f"{'phase':<22}{'CPU issue':>11}{'GPU exec':>10}   note")
        for phase_name in self.phase_names:
            host_ms = self.host_seconds[phase_name] / self.batch_count * 1000
            cuda_ms = (
                sum(start.elapsed_time(end) for start, end in self.cuda_events[phase_name])
                / self.batch_count
            )
            note = "HOST-BOUND" if host_ms > cuda_ms * 1.15 and host_ms > 0.3 else ""
            print(f"{phase_name:<22}{host_ms:>11.3f}{cuda_ms:>10.3f}   {note}")
        print(f"{'wall per batch':<22}{wall_seconds / self.batch_count * 1000:>11.3f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an EGNN model on QM9")

    training = parser.add_argument_group("training")
    training.add_argument("--exp-name", "--exp_name", default="exp_1")
    training.add_argument("--batch-size", "--batch_size", type=int, default=96)
    training.add_argument("--epochs", type=int, default=1)
    training.add_argument("--no-cuda", "--no_cuda", action="store_true")
    training.add_argument("--seed", type=int, default=1)
    training.add_argument("--log-interval", "--log_interval", type=int, default=20)
    training.add_argument("--test-interval", "--test_interval", type=int, default=1)
    training.add_argument("--outf", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    training.add_argument("--lr", type=float, default=1e-3)
    training.add_argument("--weight-decay", "--weight_decay", type=float, default=1e-16)
    training.add_argument("--property", default="homo")
    training.add_argument("--num-workers", "--num_workers", type=int, default=0)
    training.add_argument("--train-fraction", "--train_fraction", type=float, default=1.0)

    model = parser.add_argument_group("model")
    model.add_argument("--nf", type=int, default=128)
    model.add_argument("--nf-sparse", "--nf_sparse", type=int)
    model.add_argument("--attention", type=int, default=1)
    model.add_argument("--n-layers", "--n_layers", type=int, default=10)
    model.add_argument("--charge-power", "--charge_power", type=int, default=2)
    model.add_argument("--dataset-paper", "--dataset_paper", default="cormorant")
    model.add_argument("--node-attr", "--node_attr", type=int, default=0)
    model.add_argument("--pairwise", action="store_true", help="Use PairwiseEGNN")
    model.add_argument("--sparse", action="store_true", help="Use SparseEGNN")
    model.add_argument("--hybrid", action="store_true", help="Use HybridEGNN")
    model.add_argument(
        "--pairwise-layer-type",
        "--pairwise_layer_type",
        choices=("sym_asym", "egcl", "symmetric", "joint"),
        default="sym_asym",
    )
    model.add_argument("--n-standard-layers", "--n_standard_layers", type=int, default=5)
    model.add_argument("--n-pairwise-layers", "--n_pairwise_layers", type=int, default=5)

    scheduling = parser.add_argument_group("sparse edge scheduling")
    scheduling.add_argument(
        "--frame-ordering",
        "--frame_ordering",
        choices=(
            "sort_repeat",
            "sort_repeat_desc",
            "half_repeat",
            "sandwich_atomic",
            "sandwich_mass",
            "sandwich_mass_noh",
            "sandwich_penalized_h",
        ),
        default="sort_repeat",
    )
    scheduling.add_argument(
        "--frame-scoring",
        "--frame_scoring",
        choices=("atomic_number", "mass", "mass_noh", "penalized_h", "mass_product"),
        default="atomic_number",
    )
    scheduling.add_argument("--use-vizing-coloring", "--use_vizing_coloring", action="store_true")

    performance = parser.add_argument_group("performance")
    performance.add_argument(
        "--amp", action="store_true", help="Use bfloat16 autocast for forward and backward"
    )
    performance.add_argument(
        "--tf32", action="store_true", help="Allow TF32 tensor-core matrix multiplications"
    )
    performance.add_argument(
        "--fused-adam", "--fused_adam", action="store_true", help="Use fused Adam"
    )
    performance.add_argument(
        "--fused-dense",
        "--fused_dense",
        action="store_true",
        help="Use the fused dense EGNN CUDA extension",
    )
    performance.add_argument(
        "--fused-pairwise",
        "--fused_pairwise",
        action="store_true",
        help="Use the fused symmetric/asymmetric pairwise CUDA extension",
    )
    performance.add_argument(
        "--cuda-graphs",
        "--cuda_graphs",
        action="store_true",
        help="Use bounded shape-bucketed CUDA graphs for HyEGNN training",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if sum((args.pairwise, args.sparse, args.hybrid)) > 1:
        parser.error("only one of --pairwise, --sparse, and --hybrid may be selected")
    if not 0 < args.train_fraction <= 1:
        parser.error("--train-fraction must be in the interval (0, 1]")

    positive_values = {
        "--batch-size": args.batch_size,
        "--epochs": args.epochs,
        "--log-interval": args.log_interval,
        "--test-interval": args.test_interval,
        "--nf": args.nf,
        "--n-layers": args.n_layers,
    }
    if args.nf_sparse is not None:
        positive_values["--nf-sparse"] = args.nf_sparse
    for flag, value in positive_values.items():
        if value <= 0:
            parser.error(f"{flag} must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.n_standard_layers < 0 or args.n_pairwise_layers < 0:
        parser.error("hybrid layer counts cannot be negative")
    if args.hybrid and args.n_standard_layers + args.n_pairwise_layers == 0:
        parser.error("a hybrid model must contain at least one layer")

    cuda_enabled = torch.cuda.is_available() and not args.no_cuda
    if args.amp and not cuda_enabled:
        parser.error("--amp requires CUDA")
    if (args.fused_dense or args.fused_pairwise or args.fused_adam) and not cuda_enabled:
        parser.error("fused CUDA options require CUDA")
    if args.fused_dense and args.pairwise:
        parser.error("--fused-dense is not applicable to PairwiseEGNN")
    if args.fused_pairwise and not (
        args.hybrid and args.n_pairwise_layers > 0 and args.pairwise_layer_type == "sym_asym"
    ):
        parser.error("--fused-pairwise requires HybridEGNN with sym_asym layers")
    if args.cuda_graphs and not (
        cuda_enabled
        and args.hybrid
        and args.fused_dense
        and args.fused_pairwise
        and args.fused_adam
        and args.pairwise_layer_type == "sym_asym"
    ):
        parser.error(
            "--cuda-graphs requires CUDA, HybridEGNN, --fused-dense, "
            "--fused-pairwise, --fused-adam, and sym_asym layers"
        )


def build_model(args: argparse.Namespace, device: torch.device) -> ModelInfo:
    uses_sparse_edges = args.pairwise or args.sparse or args.hybrid
    sparse_layer_count = (
        args.n_standard_layers + args.n_pairwise_layers if args.hybrid else args.n_layers
    )
    pairwise_features = args.nf_sparse if args.nf_sparse is not None else args.nf

    shared_options = {
        "in_node_nf": 15,
        "in_edge_nf": 0,
        "device": device,
        "coords_weight": 1.0,
        "attention": args.attention,
        "node_attr": args.node_attr,
    }

    if args.pairwise:
        model = PairwiseEGNN(
            hidden_nf=pairwise_features,
            n_layers=args.n_layers,
            frame_ordering=args.frame_ordering,
            frame_scoring=args.frame_scoring,
            **shared_options,
        )
        name = "PairwiseEGNN"
        hidden_features = pairwise_features
        description = (
            f"Model: {name} | ordering={args.frame_ordering} | "
            f"scoring={args.frame_scoring} | hidden_nf={hidden_features} | "
            f"layers={args.n_layers}"
        )
    elif args.sparse:
        model = SparseEGNN(
            hidden_nf=pairwise_features,
            n_layers=args.n_layers,
            frame_ordering=args.frame_ordering,
            frame_scoring=args.frame_scoring,
            **shared_options,
        )
        name = "SparseEGNN"
        hidden_features = pairwise_features
        description = (
            f"Model: {name} | ordering={args.frame_ordering} | "
            f"scoring={args.frame_scoring} | hidden_nf={hidden_features} | "
            f"layers={args.n_layers}"
        )
    elif args.hybrid:
        model = HybridEGNN(
            hidden_nf=args.nf,
            pairwise_nf=pairwise_features,
            n_standard_layers=args.n_standard_layers,
            n_pairwise_layers=args.n_pairwise_layers,
            frame_ordering=args.frame_ordering,
            frame_scoring=args.frame_scoring,
            pairwise_layer_type=args.pairwise_layer_type,
            **shared_options,
        )
        name = "HybridEGNN"
        hidden_features = args.nf
        description = (
            f"Model: {name} | standard_layers={args.n_standard_layers} | "
            f"pairwise_layer_type={args.pairwise_layer_type} | "
            f"pairwise_layers={args.n_pairwise_layers} | total={sparse_layer_count} | "
            f"ordering={args.frame_ordering} | scoring={args.frame_scoring} | "
            f"hidden_nf={args.nf} | pairwise_nf={pairwise_features}"
        )
    else:
        model = EGNN(hidden_nf=args.nf, n_layers=args.n_layers, **shared_options)
        name = "EGNN"
        hidden_features = args.nf
        pairwise_features = None
        description = f"Model: {name} | layers={args.n_layers} | hidden_nf={args.nf}"

    if uses_sparse_edges:
        description += f" | vizing_coloring={args.use_vizing_coloring}"
    print(description)
    return ModelInfo(
        model=model,
        name=name,
        hidden_features=hidden_features,
        pairwise_features=pairwise_features,
        sparse_layer_count=sparse_layer_count,
        uses_sparse_edges=uses_sparse_edges,
    )


def install_fused_layers(args: argparse.Namespace, model_info: ModelInfo) -> None:
    if not (args.fused_dense or args.fused_pairwise):
        return

    source_root = str(REPOSITORY_ROOT / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)

    if args.fused_dense:
        from EGNN import FusedEGCLMask

        compressed_edges = "EGNN_BASELINE" not in os.environ
        dense_layer_count = args.n_standard_layers if args.hybrid else args.n_layers
        for layer_index in range(dense_layer_count):
            layer_name = f"gcl_{layer_index}"
            eager_layer = model_info.model._modules[layer_name]
            model_info.model._modules[layer_name] = (
                FusedEGCLMask(
                    model_info.hidden_features,
                    model_info.hidden_features,
                    model_info.hidden_features,
                    attention=bool(args.attention),
                    mask_is_ones=compressed_edges,
                )
                .to(next(model_info.model.parameters()).device)
                .load_from(eager_layer)
            )
        print(f"Using fused dense CUDA layers for {dense_layer_count} layers")

    if args.fused_pairwise:
        from HyEGNN import FusedPairwiseSymAsymLayer
        from HyEGNN.fused_pairwise_mlp import FusedPairwiseSymAsymMLP

        assert model_info.pairwise_features is not None
        fused_class = (
            FusedPairwiseSymAsymMLP
            if model_info.pairwise_features == 64
            else FusedPairwiseSymAsymLayer
        )
        device = next(model_info.model.parameters()).device
        for layer_index in range(args.n_pairwise_layers):
            layer_name = f"pairwise_{layer_index}"
            eager_layer = model_info.model._modules[layer_name]
            model_info.model._modules[layer_name] = (
                fused_class(model_info.pairwise_features).to(device).load_from(eager_layer)
            )
        print(f"Using {fused_class.__name__} for {args.n_pairwise_layers} pairwise layers")


def parameter_counts(model: nn.Module) -> tuple[int, int, dict[str, int]]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    per_layer = {
        name: count
        for name, module in model.named_modules()
        if (count := sum(parameter.numel() for parameter in module.parameters(recurse=False)))
    }
    return total, trainable, per_layer


def build_architecture_record(
    args: argparse.Namespace,
    model_info: ModelInfo,
    total_parameters: int,
    trainable_parameters: int,
    layer_parameters: dict[str, int],
    precompute_seconds: float,
    graph_runner: BucketedCudaGraphRunner | None,
) -> dict[str, Any]:
    record = {
        "model_name": model_info.name,
        "n_layers": args.n_layers,
        "hidden_nf": model_info.hidden_features,
        "in_node_nf": 15,
        "in_edge_nf": 0,
        "attention": args.attention,
        "node_attr": args.node_attr,
        "total_params": total_parameters,
        "trainable_params": trainable_parameters,
        "layer_params": layer_parameters,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "property": args.property,
        "charge_power": args.charge_power,
        "dataset_paper": args.dataset_paper,
        "seed": args.seed,
        "pairwise": args.pairwise,
        "sparse": args.sparse,
        "hybrid": args.hybrid,
        "pairwise_layer_type": args.pairwise_layer_type,
        "frame_ordering": args.frame_ordering,
        "frame_scoring": args.frame_scoring,
        "use_vizing_coloring": args.use_vizing_coloring,
        "train_fraction": args.train_fraction,
        "amp": args.amp,
        "tf32": args.tf32,
        "fused_dense": args.fused_dense,
        "fused_pairwise": args.fused_pairwise,
        "fused_adam": args.fused_adam,
        "cuda_graphs": args.cuda_graphs,
    }
    if args.hybrid:
        record.update(
            {
                "n_standard_layers": args.n_standard_layers,
                "n_pairwise_layers": args.n_pairwise_layers,
                "n_sparse_layers": model_info.sparse_layer_count,
            }
        )
    if model_info.uses_sparse_edges:
        record["precompute_time"] = precompute_seconds
    if graph_runner is not None:
        record["cuda_graph_dense_quantum"] = graph_runner.dense_quantum
        record["cuda_graph_max_buckets"] = graph_runner.max_buckets
    return record


def create_results(precompute_seconds: float, use_sparse: bool, use_graphs: bool) -> dict:
    results = {
        "epochs": [],
        "losess": [],  # Historical spelling retained for result-file compatibility.
        "test_mae": [],
        "test_mse": [],
        "test_rel_mse": [],
        "val_mae": [],
        "val_mse": [],
        "val_rel_mse": [],
        "best_val": 1e10,
        "best_test": 1e10,
        "best_epoch": 0,
        "best_test_mae": 1e10,
        "best_test_mse": 1e10,
        "best_test_rel_mse": 1e10,
        "train_mse": [],
        "best_train_mse": 1e10,
        "best_train_mse_epoch": 0,
        "best_train_mse_time": 0.0,
        "train_time_total": 0.0,
        "train_time_per_epoch": [],
        "validation_time_total": 0.0,
        "test_time_total": 0.0,
        "total_time": 0.0,
    }
    if use_sparse:
        results["precompute_time"] = precompute_seconds
    if use_graphs:
        results["cuda_graph_captures"] = 0
        results["cuda_graph_eager_fallbacks"] = 0
    return results


def run_epoch(
    epoch: int, loader: DataLoader, partition: str, context: TrainingContext
) -> tuple[float, float, float, float]:
    args = context.args
    model = context.model_info.model
    is_training = partition == "train"
    uses_graph = is_training and context.graph_runner is not None
    model.train(is_training)

    loss_values: list[Tensor] = []
    sample_count = 0
    accumulators = {
        name: torch.zeros((), device=context.device)
        for name in ("loss", "mae", "mse", "relative_mse")
    }
    profiler = (
        PhaseProfiler(
            [
                "loader_wait",
                "h2d_prep",
                "edges",
                "forward",
                "loss",
                "backward",
                "optimizer",
                "metrics",
            ]
        )
        if PROFILE_BATCH_COUNT and is_training and epoch == 0 and context.device.type == "cuda"
        else None
    )
    previous_batch_end = time.perf_counter()
    profile_wall_start = previous_batch_end

    for batch_index, batch in enumerate(loader):
        if profiler is not None:
            profiler.host_seconds["loader_wait"] += time.perf_counter() - previous_batch_end
            profiler.start("h2d_prep")
        if is_training and not uses_graph:
            context.optimizer.zero_grad()

        batch_size, node_count, _ = batch["positions"].size()
        positions = (
            batch["positions"]
            .view(batch_size * node_count, -1)
            .to(context.device, torch.float32, non_blocking=True)
        )
        node_mask = (
            batch["atom_mask"]
            .view(batch_size * node_count, -1)
            .to(context.device, torch.float32, non_blocking=True)
        )
        one_hot = batch["one_hot"].to(context.device, torch.float32, non_blocking=True)
        charges = batch["charges"].to(context.device, torch.float32, non_blocking=True)
        nodes = qm9_utils.preprocess_input(
            one_hot, charges, args.charge_power, context.charge_scale, context.device
        ).view(batch_size * node_count, -1)

        if "dense_rows" in batch:
            edges = (
                batch["dense_rows"].to(context.device, non_blocking=True),
                batch["dense_cols"].to(context.device, non_blocking=True),
            )
            dense_edge_mask = (
                None if args.fused_dense else torch.ones(edges[0].size(0), 1, device=context.device)
            )
        else:
            edges = tuple(qm9_utils.get_adj_matrix(node_count, batch_size, context.device))
            dense_edge_mask = batch["edge_mask"].to(
                context.device, torch.float32, non_blocking=True
            )
        labels = batch[args.property].to(context.device, torch.float32, non_blocking=True)

        if profiler is not None:
            profiler.stop()
            profiler.start("edges")

        sparse_edges = None
        if context.model_info.uses_sparse_edges and context.coloring_cache is not None:
            if "sparse_rows" in batch:
                sparse_edges = [
                    (
                        rows.to(context.device, non_blocking=True),
                        columns.to(context.device, non_blocking=True),
                        None,
                    )
                    for rows, columns in zip(batch["sparse_rows"], batch["sparse_cols"])
                ]
            else:
                sparse_edges = assemble_batch_sparse_edges(
                    coloring_cache=context.coloring_cache,
                    charges_batch=batch["charges"],
                    atom_mask_batch=batch["atom_mask"],
                    n_nodes=node_count,
                    n_layers=context.model_info.sparse_layer_count,
                    device=context.device,
                )

        if profiler is not None:
            profiler.stop()
            profiler.start("forward")

        if uses_graph:
            assert context.graph_runner is not None
            loss, real_predictions = context.graph_runner.run_training_step(
                nodes,
                positions,
                node_mask,
                edges,
                dense_edge_mask,
                labels,
                node_count,
                sparse_edges,
            )
            if profiler is not None:
                profiler.stop()
        else:
            with (
                torch.set_grad_enabled(is_training),
                torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp),
            ):
                predictions = model(
                    h0=nodes,
                    x=positions,
                    edges=edges,
                    edge_attr=None,
                    node_mask=node_mask,
                    edge_mask=dense_edge_mask,
                    n_nodes=node_count,
                    sparse_edges_per_layer=sparse_edges,
                ).float()
            if profiler is not None:
                profiler.stop()

            real_predictions = context.target_deviation * predictions + context.target_mean
            if is_training:
                if profiler is not None:
                    profiler.start("loss")
                loss = context.loss_function(
                    predictions,
                    (labels - context.target_mean) / context.target_deviation,
                )
                if profiler is not None:
                    profiler.stop()
                    profiler.start("backward")
                loss.backward()
                if profiler is not None:
                    profiler.stop()
                    profiler.start("optimizer")
                context.optimizer.step()
                if profiler is not None:
                    profiler.stop()
            else:
                loss = context.loss_function(real_predictions, labels)

        if profiler is not None:
            profiler.start("metrics")
        sample_count += batch_size
        with torch.no_grad():
            detached_loss = loss.detach()
            prediction_error = real_predictions - labels
            accumulators["loss"] += detached_loss * batch_size
            accumulators["mae"] += prediction_error.abs().sum()
            batch_mse = prediction_error.square().sum()
            accumulators["mse"] += batch_mse
            accumulators["relative_mse"] += (
                batch_mse / (labels.square().sum() + 1e-10)
            ) * batch_size
            # Graph buckets reuse one output address, so clone values that a
            # later replay would otherwise overwrite.
            loss_values.append(detached_loss.clone() if uses_graph else detached_loss)

        prefix = "" if is_training else f">> {partition}\t"
        if batch_index % args.log_interval == 0:
            recent_loss = torch.stack(loss_values[-10:]).mean().item()
            print(f"{prefix}Epoch {epoch} \t Iteration {batch_index} \t loss {recent_loss:.4f}")

        if profiler is not None:
            profiler.stop()
            profiler.batch_count += 1
            previous_batch_end = time.perf_counter()
            if profiler.batch_count == PROFILE_BATCH_COUNT:
                profiler.report(time.perf_counter() - profile_wall_start)
                profiler = None

    if is_training:
        context.scheduler.step()

    totals = tuple(accumulators[name].item() for name in ("loss", "mae", "mse", "relative_mse"))
    return tuple(total / sample_count for total in totals)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_epoch(
    epoch: int, loader: DataLoader, partition: str, context: TrainingContext
) -> tuple[tuple[float, float, float, float], float]:
    synchronize(context.device)
    start_time = time.perf_counter()
    metrics = run_epoch(epoch, loader, partition, context)
    synchronize(context.device)
    return metrics, time.perf_counter() - start_time


def run_training(
    dataloaders: dict[str, DataLoader],
    context: TrainingContext,
    output_directory: Path,
    precompute_seconds: float,
) -> None:
    args = context.args
    results = create_results(
        precompute_seconds,
        context.model_info.uses_sparse_edges,
        context.graph_runner is not None,
    )
    result_path = output_directory / "losess.json"
    minimum_mse_improvement = 0.00005

    synchronize(context.device)
    total_start_time = time.perf_counter()

    for epoch in range(args.epochs):
        train_metrics, training_seconds = timed_epoch(epoch, dataloaders["train"], "train", context)
        _, _, train_mse, _ = train_metrics
        results["train_time_total"] += training_seconds
        results["train_time_per_epoch"].append(training_seconds)
        results["train_mse"].append(train_mse)

        if train_mse < results["best_train_mse"] - minimum_mse_improvement:
            results["best_train_mse"] = train_mse
            results["best_train_mse_epoch"] = epoch
            results["best_train_mse_time"] = results["train_time_total"]

        if epoch % args.test_interval == 0:
            validation_metrics, validation_seconds = timed_epoch(
                epoch, dataloaders["valid"], "valid", context
            )
            test_metrics, test_seconds = timed_epoch(epoch, dataloaders["test"], "test", context)
            validation_loss, validation_mae, validation_mse, validation_relative_mse = (
                validation_metrics
            )
            test_loss, test_mae, test_mse, test_relative_mse = test_metrics

            results["validation_time_total"] += validation_seconds
            results["test_time_total"] += test_seconds
            results["epochs"].append(epoch)
            results["losess"].append(test_loss)
            results["test_mae"].append(test_mae)
            results["test_mse"].append(test_mse)
            results["test_rel_mse"].append(test_relative_mse)
            results["val_mae"].append(validation_mae)
            results["val_mse"].append(validation_mse)
            results["val_rel_mse"].append(validation_relative_mse)

            if validation_loss < results["best_val"]:
                results["best_val"] = validation_loss
                results["best_test"] = test_loss
                results["best_epoch"] = epoch
                results["best_test_mae"] = test_mae
                results["best_test_mse"] = test_mse
                results["best_test_rel_mse"] = test_relative_mse

            print(f"Val loss: {validation_loss:.4f} \t test loss: {test_loss:.4f} \t epoch {epoch}")
            print(
                f"  Train MSE: {train_mse:.6f} \t Test MAE: {test_mae:.6f} "
                f"\t MSE: {test_mse:.6f} \t RelMSE: {test_relative_mse:.6f}"
            )
            print(
                f"Best: val loss: {results['best_val']:.4f} \t "
                f"test loss: {results['best_test']:.4f} \t epoch {results['best_epoch']}"
            )
            print(
                f"Best train MSE: {results['best_train_mse']:.6f} "
                f"\t epoch {results['best_train_mse_epoch']}"
            )
            print(
                f"  Train time: {results['train_time_total']:.1f}s \t "
                f"Validation time: {results['validation_time_total']:.1f}s \t "
                f"Test time: {results['test_time_total']:.1f}s"
            )
            if context.model_info.uses_sparse_edges:
                print(f"  Precompute time (one-time): {precompute_seconds:.1f}s")

        synchronize(context.device)
        results["total_time"] = time.perf_counter() - total_start_time
        if context.graph_runner is not None:
            results["cuda_graph_captures"] = context.graph_runner.capture_count
            results["cuda_graph_eager_fallbacks"] = context.graph_runner.eager_fallback_count
        with result_path.open("w", encoding="utf-8") as output_file:
            json.dump(results, output_file, indent=4)


def run_experiment(
    args: argparse.Namespace,
    dataloaders: dict[str, DataLoader],
    charge_scale: Any,
    device: torch.device,
) -> None:
    if args.train_fraction < 1:
        original_loader = dataloaders["train"]
        full_training_dataset = original_loader.dataset
        subset_size = max(1, int(len(full_training_dataset) * args.train_fraction))
        training_subset = Subset(full_training_dataset, range(subset_size))
        dataset.shutdown_dataloaders({"train": original_loader})
        dataloaders["train"] = dataset.create_dataloader(
            training_subset, "train", args.batch_size, args.num_workers
        )
        print(
            f"Using {subset_size}/{len(full_training_dataset)} training samples "
            f"({args.train_fraction:.2%})"
        )

    target_mean, target_deviation = qm9_utils.compute_mean_mad(dataloaders, args.property)
    model_info = build_model(args, device)
    install_fused_layers(args, model_info)
    if args.cuda_graphs:
        replace_linear_layers_for_graph_capture(model_info.model)
        print("Using graph-safe GEMM and bias linear layers")

    total_parameters, trainable_parameters, layer_parameters = parameter_counts(model_info.model)
    print(f"Parameters: {total_parameters:,} (trainable: {trainable_parameters:,})")
    print(f"Per-layer params: {json.dumps(layer_parameters, indent=2)}")

    coloring_cache = None
    precompute_seconds = 0.0
    if model_info.uses_sparse_edges:
        print("Precomputing molecule colorings (one-time cost)...")
        precompute_start = time.perf_counter()
        coloring_cache = precompute_molecule_colorings(
            dataloaders=dataloaders,
            n_layers=model_info.sparse_layer_count,
            frame_ordering=args.frame_ordering,
            frame_scoring=args.frame_scoring,
            use_vizing_coloring=args.use_vizing_coloring,
        )
        precompute_seconds = time.perf_counter() - precompute_start
        print(f"Precompute time: {precompute_seconds:.2f}s")

        if args.num_workers > 0:
            skip_first = args.n_standard_layers if args.hybrid else 0
            rebuilt_dataloaders = dataset.rebuild_dataloaders(
                dataloaders,
                args.batch_size,
                args.num_workers,
                SparseEdgeCollate(
                    base_collate,
                    coloring_cache,
                    model_info.sparse_layer_count,
                    skip_first=skip_first,
                ),
            )
            dataloaders.clear()
            dataloaders.update(rebuilt_dataloaders)
            print(f"Sparse edge assembly moved into {args.num_workers} DataLoader workers")

    optimizer_learning_rate = torch.tensor(args.lr, device=device) if args.cuda_graphs else args.lr
    optimizer = optim.Adam(
        model_info.model.parameters(),
        lr=optimizer_learning_rate,
        weight_decay=args.weight_decay,
        fused=args.fused_adam,
        capturable=args.cuda_graphs,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    loss_function = nn.L1Loss()
    graph_runner = (
        BucketedCudaGraphRunner(
            model=model_info.model,
            optimizer=optimizer,
            loss_function=loss_function,
            mean=target_mean,
            mean_absolute_deviation=target_deviation,
            sparse_layer_count=model_info.sparse_layer_count,
            sparse_start=args.n_standard_layers,
            amp=args.amp,
        )
        if args.cuda_graphs
        else None
    )
    if graph_runner is not None:
        print("Using bucketed CUDA graphs for complete HyEGNN training steps")

    output_directory = args.outf / args.exp_name
    output_directory.mkdir(parents=True, exist_ok=True)
    architecture = build_architecture_record(
        args,
        model_info,
        total_parameters,
        trainable_parameters,
        layer_parameters,
        precompute_seconds,
        graph_runner,
    )
    with (output_directory / "architecture.json").open("w", encoding="utf-8") as architecture_file:
        json.dump(architecture, architecture_file, indent=4)

    context = TrainingContext(
        args=args,
        model_info=model_info,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_function=loss_function,
        device=device,
        charge_scale=charge_scale,
        target_mean=target_mean,
        target_deviation=target_deviation,
        coloring_cache=coloring_cache,
        graph_runner=graph_runner,
    )
    run_training(
        dataloaders,
        context,
        output_directory,
        precompute_seconds,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda" if args.cuda else "cpu")
    torch.manual_seed(args.seed)
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    args.outf.mkdir(parents=True, exist_ok=True)
    dataloaders, charge_scale = dataset.retrieve_dataloaders(args.batch_size, args.num_workers)
    try:
        run_experiment(args, dataloaders, charge_scale, device)
    finally:
        dataset.shutdown_dataloaders(dataloaders)


if __name__ == "__main__":
    main()

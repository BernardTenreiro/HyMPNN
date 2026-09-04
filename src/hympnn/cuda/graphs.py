"""Bucketed CUDA graph execution for EGNN and HybridEGNN training.

QM9 batches have a small set of node counts but variable dense and sparse edge
counts. Each bucket preserves the real batch, appends one masked scratch graph,
and pads edge lists into that graph. The captured region contains the forward
pass, loss, backward pass, gradient clearing, and fused Adam step.

Input staging and metrics stay outside the graph. Validation and test remain
eager so training buckets do not compete with a second graph cache.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional

EdgeIndex = tuple[Tensor, Tensor]
SparseLayerEdges = tuple[Tensor, Tensor, Tensor | None]


def _round_up(value: int, quantum: int) -> int:
    return ((value + quantum - 1) // quantum) * quantum


def _read_capacity(
    name: str,
    default: int,
    minimum: int = 1,
    legacy_name: str | None = None,
) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None and legacy_name is not None:
        raw_value = os.environ.get(legacy_name)
    value = int(default if raw_value is None else raw_value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, received {value}")
    return value


class _GraphSafeLinear(nn.Module):
    """Linear layer that avoids a shape-specific cuBLASLt bias epilogue.

    cuBLASLt can invalidate capture when the same biased weight is recorded at
    several M dimensions in one graph pool. A plain GEMM followed by bias is
    stable across buckets, and graph replay hides the additional add launch.
    Parameter objects and state-dict names are preserved.
    """

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias

    def forward(self, value: Tensor) -> Tensor:
        value = functional.linear(value, self.weight, None)
        return value if self.bias is None else value + self.bias


def replace_linear_layers_for_graph_capture(module: nn.Module) -> None:
    """Recursively replace biased linear calls without reallocating parameters."""
    for name, child in tuple(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, _GraphSafeLinear(child))
        else:
            replace_linear_layers_for_graph_capture(child)


@dataclass(frozen=True)
class _GraphBucketKey:
    batch_size: int
    node_count: int
    dense_capacity: int
    sparse_capacities: tuple[int, ...]


@dataclass
class _GraphBucket:
    key: _GraphBucketKey
    scratch_node: int
    nodes: Tensor
    positions: Tensor
    node_mask: Tensor
    labels: Tensor
    dense_rows: Tensor
    dense_columns: Tensor
    dense_mask: Tensor
    node_ids: Tensor
    dense_offsets: Tensor
    sparse_edges: list[SparseLayerEdges] | None
    graph: torch.cuda.CUDAGraph
    loss: Tensor | None = None
    predictions: Tensor | None = None


class BucketedCudaGraphRunner:
    """Lazily capture and replay fixed-capacity model training steps."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_function: nn.Module,
        mean: Tensor | float,
        mean_absolute_deviation: Tensor | float,
        sparse_layer_count: int,
        sparse_start: int = 0,
        amp: bool = False,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("BucketedCudaGraphRunner requires CUDA")
        if not all(group.get("capturable", False) for group in optimizer.param_groups):
            raise ValueError("the optimizer must be constructed with capturable=True")

        first_parameter = next(model.parameters())
        if first_parameter.device.type != "cuda":
            raise ValueError("the model must be on a CUDA device")

        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = first_parameter.device
        self.mean = torch.as_tensor(mean, device=self.device, dtype=torch.float32)
        self.mean_absolute_deviation = torch.as_tensor(
            mean_absolute_deviation, device=self.device, dtype=torch.float32
        )
        self.sparse_layer_count = sparse_layer_count
        self.sparse_start = sparse_start
        self.amp = amp
        self.buckets: dict[_GraphBucketKey, _GraphBucket] = {}
        self.capture_count = 0
        self.eager_fallback_count = 0
        self._reported_fallbacks: set[_GraphBucketKey] = set()

        # Replays are serialized and their outputs are consumed before the next
        # replay, so all buckets can safely share one private memory pool.
        self._graph_pool = torch.cuda.graph_pool_handle()

        # Tight dense buckets minimize padding. Capture is bounded to common
        # shapes; the rest safely use the eager overflow path. Environment
        # overrides allow capacity studies without editing the source.
        self.dense_base = _read_capacity(
            "HYMPNN_CUDA_GRAPH_DENSE_CAP", 0, minimum=0, legacy_name="EGNN_GRAPH_DENSE_CAP"
        )
        self.dense_quantum = _read_capacity(
            "HYMPNN_CUDA_GRAPH_DENSE_QUANTUM",
            4096,
            legacy_name="EGNN_GRAPH_DENSE_QUANTUM",
        )
        self.sparse_quantum = _read_capacity(
            "HYMPNN_CUDA_GRAPH_SPARSE_QUANTUM",
            128,
            legacy_name="EGNN_GRAPH_SPARSE_QUANTUM",
        )
        self.sparse_small = _read_capacity(
            "HYMPNN_CUDA_GRAPH_SPARSE_SMALL_CAP",
            768,
            legacy_name="EGNN_GRAPH_SPARSE_SMALL_CAP",
        )
        self.sparse_large = _read_capacity(
            "HYMPNN_CUDA_GRAPH_SPARSE_LARGE_CAP",
            1024,
            legacy_name="EGNN_GRAPH_SPARSE_LARGE_CAP",
        )
        self.max_buckets = _read_capacity(
            "HYMPNN_CUDA_GRAPH_MAX_BUCKETS", 7, legacy_name="EGNN_GRAPH_MAX_BUCKETS"
        )

        # Graph replay requires fixed gradient and optimizer-state addresses.
        # Initialize both without taking a real optimization step.
        for group in self.optimizer.param_groups:
            for parameter in group["params"]:
                if not parameter.requires_grad:
                    continue
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                state = self.optimizer.state[parameter]
                if state:
                    continue
                state["step"] = torch.zeros((), device=parameter.device, dtype=torch.float32)
                state["exp_avg"] = torch.zeros_like(parameter)
                state["exp_avg_sq"] = torch.zeros_like(parameter)
                if group.get("amsgrad", False):
                    state["max_exp_avg_sq"] = torch.zeros_like(parameter)

    def _sparse_base_capacity(self, ordinal: int) -> int:
        active_layer_count = self.sparse_layer_count - self.sparse_start
        if active_layer_count >= 2 and ordinal >= active_layer_count - 2:
            return self.sparse_large
        return self.sparse_small

    def _bucket_key(
        self,
        batch_size: int,
        node_count: int,
        edges: EdgeIndex,
        sparse_edges: Sequence[SparseLayerEdges] | None,
    ) -> _GraphBucketKey:
        dense_count = int(edges[0].numel())
        dense_capacity = max(self.dense_base, _round_up(dense_count, self.dense_quantum))

        capacities = []
        if self.sparse_layer_count:
            if sparse_edges is None or len(sparse_edges) != self.sparse_layer_count:
                raise ValueError("sparse edge list does not match the configured layer count")
            for ordinal, layer in enumerate(range(self.sparse_start, self.sparse_layer_count)):
                edge_count = int(sparse_edges[layer][0].numel())
                capacities.append(
                    max(
                        self._sparse_base_capacity(ordinal),
                        _round_up(edge_count, self.sparse_quantum),
                    )
                )

        return _GraphBucketKey(int(batch_size), int(node_count), dense_capacity, tuple(capacities))

    def _model_call(self, bucket: _GraphBucket) -> Tensor:
        predictions = self.model(
            h0=bucket.nodes,
            x=bucket.positions,
            edges=(bucket.dense_rows, bucket.dense_columns),
            edge_attr=None,
            node_mask=bucket.node_mask,
            edge_mask=bucket.dense_mask,
            n_nodes=bucket.key.node_count,
            sparse_edges_per_layer=bucket.sparse_edges,
        )[: bucket.key.batch_size]
        return predictions.float()

    def _forward_loss(self, bucket: _GraphBucket) -> tuple[Tensor, Tensor]:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            predictions = self._model_call(bucket)
            predictions_real = self.mean_absolute_deviation * predictions + self.mean
            normalized_labels = (bucket.labels - self.mean) / self.mean_absolute_deviation
            loss = self.loss_function(predictions, normalized_labels)
        return loss, predictions_real

    def _run_eager(
        self,
        nodes: Tensor,
        positions: Tensor,
        node_mask: Tensor,
        edges: EdgeIndex,
        dense_edge_mask: Tensor | None,
        labels: Tensor,
        node_count: int,
        sparse_edges: Sequence[SparseLayerEdges] | None,
    ) -> tuple[Tensor, Tensor]:
        """Execute an overflow shape after the bounded graph cache is full."""
        self.optimizer.zero_grad(set_to_none=False)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            predictions = self.model(
                h0=nodes,
                x=positions,
                edges=edges,
                edge_attr=None,
                node_mask=node_mask,
                edge_mask=dense_edge_mask,
                n_nodes=node_count,
                sparse_edges_per_layer=sparse_edges,
            ).float()
            predictions_real = self.mean_absolute_deviation * predictions + self.mean
            loss = self.loss_function(
                predictions, (labels - self.mean) / self.mean_absolute_deviation
            )
        loss.backward()
        self.optimizer.step()
        return loss, predictions_real

    def _snapshot_training_state(
        self,
    ) -> tuple[list[tuple[Tensor, Tensor]], list[tuple[Tensor, Tensor]]]:
        parameters = [
            (parameter, parameter.detach().clone()) for parameter in self.model.parameters()
        ]
        optimizer_tensors = []
        for state in self.optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    optimizer_tensors.append((value, value.detach().clone()))
        return parameters, optimizer_tensors

    @staticmethod
    @torch.no_grad()
    def _restore_training_state(
        snapshot: tuple[list[tuple[Tensor, Tensor]], list[tuple[Tensor, Tensor]]],
    ) -> None:
        parameters, optimizer_tensors = snapshot
        for target, saved in parameters:
            target.copy_(saved)
        for target, saved in optimizer_tensors:
            target.copy_(saved)

    def _allocate_bucket(self, key: _GraphBucketKey, node_features: Tensor) -> _GraphBucket:
        # Append one masked graph. Padded dense and sparse edges land on its
        # first node and therefore cannot influence a real prediction.
        device, dtype = node_features.device, node_features.dtype
        total_nodes = (key.batch_size + 1) * key.node_count
        scratch_node = key.batch_size * key.node_count

        nodes = torch.zeros(total_nodes, node_features.size(1), device=device, dtype=dtype)
        positions = torch.zeros(total_nodes, 3, device=device, dtype=dtype)
        node_mask = torch.zeros(total_nodes, 1, device=device, dtype=dtype)
        labels = torch.empty(key.batch_size, device=device, dtype=torch.float32)
        dense_rows = torch.full(
            (key.dense_capacity,), scratch_node, device=device, dtype=torch.long
        )
        dense_columns = torch.full(
            (key.dense_capacity,), scratch_node, device=device, dtype=torch.long
        )
        dense_mask = torch.zeros(key.dense_capacity, 1, device=device, dtype=dtype)

        # Fused dense layers memoize row segment boundaries. A graph bucket
        # reuses its row tensor, so retain a stable output address and refresh
        # the values before each replay.
        from .fused_dense import row_offsets

        node_ids = torch.arange(total_nodes + 1, device=device, dtype=torch.long)
        dense_offsets = row_offsets(dense_rows, total_nodes)

        sparse_layers: list[SparseLayerEdges] = []
        capacity_iterator = iter(key.sparse_capacities)
        for layer in range(self.sparse_layer_count):
            if layer < self.sparse_start:
                row = torch.empty(0, device=device, dtype=torch.long)
                column = torch.empty(0, device=device, dtype=torch.long)
                mask = torch.empty((0, 1), device=device, dtype=dtype)
            else:
                capacity = next(capacity_iterator)
                row = torch.full((capacity,), scratch_node, device=device, dtype=torch.long)
                column = torch.full((capacity,), scratch_node, device=device, dtype=torch.long)
                mask = torch.zeros((capacity, 1), device=device, dtype=dtype)
            sparse_layers.append((row, column, mask))

        return _GraphBucket(
            key=key,
            scratch_node=scratch_node,
            nodes=nodes,
            positions=positions,
            node_mask=node_mask,
            labels=labels,
            dense_rows=dense_rows,
            dense_columns=dense_columns,
            dense_mask=dense_mask,
            node_ids=node_ids,
            dense_offsets=dense_offsets,
            sparse_edges=sparse_layers if self.sparse_layer_count else None,
            graph=torch.cuda.CUDAGraph(),
        )

    @torch.no_grad()
    def _copy_inputs(
        self,
        bucket: _GraphBucket,
        nodes: Tensor,
        positions: Tensor,
        node_mask: Tensor,
        edges: EdgeIndex,
        dense_edge_mask: Tensor | None,
        labels: Tensor,
        sparse_edges: Sequence[SparseLayerEdges] | None,
    ) -> None:
        real_node_count = bucket.key.batch_size * bucket.key.node_count
        bucket.nodes[:real_node_count].copy_(nodes)
        bucket.positions[:real_node_count].copy_(positions)
        bucket.node_mask[:real_node_count].copy_(node_mask)
        bucket.labels.copy_(labels)

        dense_count = edges[0].numel()
        bucket.dense_rows.fill_(bucket.scratch_node)
        bucket.dense_columns.fill_(bucket.scratch_node)
        bucket.dense_rows[:dense_count].copy_(edges[0])
        bucket.dense_columns[:dense_count].copy_(edges[1])
        bucket.dense_mask.zero_()
        if dense_edge_mask is None:
            bucket.dense_mask[:dense_count].fill_(1)
        else:
            bucket.dense_mask[:dense_count].copy_(dense_edge_mask)
        torch.searchsorted(bucket.dense_rows, bucket.node_ids, out=bucket.dense_offsets)

        if self.sparse_layer_count:
            assert sparse_edges is not None
            assert bucket.sparse_edges is not None
            for layer in range(self.sparse_start, self.sparse_layer_count):
                source_row, source_column, _ = sparse_edges[layer]
                target_row, target_column, target_mask = bucket.sparse_edges[layer]
                edge_count = source_row.numel()
                target_row.fill_(bucket.scratch_node)
                target_column.fill_(bucket.scratch_node)
                target_row[:edge_count].copy_(source_row)
                target_column[:edge_count].copy_(source_column)
                target_mask.zero_()
                target_mask[:edge_count].fill_(1)

    def _warm_up_training(self, bucket: _GraphBucket, iterations: int = 2) -> None:
        snapshot = self._snapshot_training_state()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(iterations):
                self.optimizer.zero_grad(set_to_none=False)
                loss, _ = self._forward_loss(bucket)
                loss.backward()
                self.optimizer.step()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self._restore_training_state(snapshot)
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.grad.zero_()
        torch.cuda.synchronize()

    def _capture_bucket(self, bucket: _GraphBucket) -> None:
        self._warm_up_training(bucket)

        # Capture executes the step once. Restore it so the caller's first
        # replay performs exactly one update, matching subsequent cache hits.
        snapshot = self._snapshot_training_state()
        with torch.cuda.graph(bucket.graph, pool=self._graph_pool):
            self.optimizer.zero_grad(set_to_none=False)
            bucket.loss, bucket.predictions = self._forward_loss(bucket)
            bucket.loss.backward()
            self.optimizer.step()
        self._restore_training_state(snapshot)
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.grad.zero_()
        torch.cuda.synchronize()

        self.capture_count += 1
        print(
            "CUDA graph captured: "
            f"bs={bucket.key.batch_size} n={bucket.key.node_count} "
            f"dense={bucket.key.dense_capacity} sparse={bucket.key.sparse_capacities}"
        )

    def run_training_step(
        self,
        nodes: Tensor,
        positions: Tensor,
        node_mask: Tensor,
        edges: EdgeIndex,
        dense_edge_mask: Tensor | None,
        labels: Tensor,
        node_count: int,
        sparse_edges: Sequence[SparseLayerEdges] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Execute one captured or eager training step for the supplied batch."""
        batch_size = labels.numel()
        key = self._bucket_key(batch_size, node_count, edges, sparse_edges)
        bucket = self.buckets.get(key)

        if bucket is None:
            # CUDA/cuBLAS graph state becomes unreliable when too many matrix
            # shapes are retained. Less common shapes execute eagerly once the
            # bounded cache is full instead of risking capture invalidation.
            if len(self.buckets) >= self.max_buckets:
                self.eager_fallback_count += 1
                if key not in self._reported_fallbacks:
                    self._reported_fallbacks.add(key)
                    print(
                        "CUDA graph cache full; using eager fallback for "
                        f"bs={key.batch_size} n={key.node_count} dense={key.dense_capacity}"
                    )
                return self._run_eager(
                    nodes,
                    positions,
                    node_mask,
                    edges,
                    dense_edge_mask,
                    labels,
                    node_count,
                    sparse_edges,
                )

            bucket = self._allocate_bucket(key, nodes)
            self.buckets[key] = bucket
            self._copy_inputs(
                bucket,
                nodes,
                positions,
                node_mask,
                edges,
                dense_edge_mask,
                labels,
                sparse_edges,
            )
            self._capture_bucket(bucket)
        else:
            self._copy_inputs(
                bucket,
                nodes,
                positions,
                node_mask,
                edges,
                dense_edge_mask,
                labels,
                sparse_edges,
            )

        bucket.graph.replay()
        assert bucket.loss is not None
        assert bucket.predictions is not None
        return bucket.loss, bucket.predictions

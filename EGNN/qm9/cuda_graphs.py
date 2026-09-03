"""Bucketed CUDA-graph execution for HybridEGNN training.

QM9 batches have a small set of node counts but variable dense and sparse
edge counts.  Each bucket preserves the real batch, appends one masked scratch
graph, and pads edge lists into that graph.  The captured region contains the
forward pass, loss, backward pass, gradient clearing, and fused Adam step.
Input staging and metrics stay outside the graph; validation and test remain
eager so training buckets do not compete with a second graph cache.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F


def _round_up(value: int, quantum: int) -> int:
    return ((value + quantum - 1) // quantum) * quantum


class _GraphSafeLinear(nn.Module):
    """Linear view that avoids cuBLASLt's shape-specific fused bias epilogue.

    cuBLASLt can invalidate capture when the same biased weight is recorded at
    several M dimensions in one graph pool.  A plain GEMM followed by bias is
    stable across buckets, and the extra add launch is hidden by graph replay.
    Parameter objects and state-dict names are preserved.
    """

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias

    def forward(self, value):
        value = F.linear(value, self.weight, None)
        return value if self.bias is None else value + self.bias


def make_linears_graph_safe(module: nn.Module) -> None:
    """Recursively replace biased Linear calls without reallocating parameters."""
    for name, child in tuple(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, _GraphSafeLinear(child))
        else:
            make_linears_graph_safe(child)


@dataclass(frozen=True)
class _BucketKey:
    batch_size: int
    n_nodes: int
    dense_capacity: int
    sparse_capacities: Tuple[int, ...]


class _Bucket:
    pass


class BucketedCUDAGraphRunner:
    """Lazily capture and replay fixed-capacity HybridEGNN training steps."""

    def __init__(self, model, optimizer, loss_fn, mean, mad,
                 expected_batch_size, n_sparse_layers, sparse_start=0,
                 amp=False):
        if not torch.cuda.is_available():
            raise RuntimeError("BucketedCUDAGraphRunner requires CUDA")
        if not all(group.get("capturable", False)
                   for group in optimizer.param_groups):
            raise ValueError("the optimizer must be constructed with capturable=True")

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.mean = torch.as_tensor(mean, device="cuda", dtype=torch.float32)
        self.mad = torch.as_tensor(mad, device="cuda", dtype=torch.float32)
        self.expected_batch_size = expected_batch_size
        self.n_sparse_layers = n_sparse_layers
        self.sparse_start = sparse_start
        self.amp = amp
        self.buckets: Dict[_BucketKey, _Bucket] = {}
        self.captures = 0
        self.eager_fallbacks = 0
        self.fallback_keys = set()

        # Graphs replay serially and their temporary results are consumed before
        # the next replay, so they can safely share one private memory pool.
        self.pool = torch.cuda.graph_pool_handle()

        # Tight 1024-edge buckets minimize dense padding.  Capture is bounded to
        # the first 16 common shapes; measured over full QM9 epochs they cover
        # about 89% of batches, and the rest safely use the eager overflow path.
        # Sparse padding is nearly free because it only touches the scratch graph.
        # Environment overrides allow capacity studies without source edits.
        self.dense_base = int(os.environ.get("EGNN_GRAPH_DENSE_CAP", "0"))
        self.dense_quantum = int(
            os.environ.get("EGNN_GRAPH_DENSE_QUANTUM", "1024"))
        self.sparse_quantum = int(
            os.environ.get("EGNN_GRAPH_SPARSE_QUANTUM", "128"))
        self.sparse_small = int(
            os.environ.get("EGNN_GRAPH_SPARSE_SMALL_CAP", "768"))
        self.sparse_large = int(
            os.environ.get("EGNN_GRAPH_SPARSE_LARGE_CAP", "1024"))
        self.max_buckets = int(
            os.environ.get("EGNN_GRAPH_MAX_BUCKETS", "16"))

        # CUDA graph replay requires fixed gradient and optimizer-state
        # addresses.  Initialize both without taking a real optimization step.
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if not param.requires_grad:
                    continue
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                state = self.optimizer.state[param]
                if state:
                    continue
                state["step"] = torch.zeros((), device=param.device,
                                            dtype=torch.float32)
                state["exp_avg"] = torch.zeros_like(param)
                state["exp_avg_sq"] = torch.zeros_like(param)
                if group.get("amsgrad", False):
                    state["max_exp_avg_sq"] = torch.zeros_like(param)

    def _sparse_base(self, ordinal: int) -> int:
        active = self.n_sparse_layers - self.sparse_start
        return (self.sparse_large
                if active >= 2 and ordinal >= active - 2
                else self.sparse_small)

    def _key(self, batch_size, n_nodes, edges, sparse_edges):
        dense_count = int(edges[0].numel())
        dense_capacity = max(
            self.dense_base, _round_up(dense_count, self.dense_quantum))

        capacities = []
        if self.n_sparse_layers:
            if sparse_edges is None or len(sparse_edges) != self.n_sparse_layers:
                raise ValueError(
                    "sparse edge list does not match the configured layer count")
            for ordinal, layer in enumerate(
                    range(self.sparse_start, self.n_sparse_layers)):
                count = int(sparse_edges[layer][0].numel())
                capacities.append(max(
                    self._sparse_base(ordinal),
                    _round_up(count, self.sparse_quantum)))

        return _BucketKey(int(batch_size), int(n_nodes), dense_capacity,
                          tuple(capacities))

    def _model_call(self, bucket):
        pred = self.model(
            h0=bucket.nodes,
            x=bucket.positions,
            edges=(bucket.dense_rows, bucket.dense_cols),
            edge_attr=None,
            node_mask=bucket.node_mask,
            edge_mask=bucket.dense_mask,
            n_nodes=bucket.key.n_nodes,
            sparse_edges_per_layer=bucket.sparse_edges,
        )[:bucket.key.batch_size]
        return pred.float()

    def _forward_loss(self, bucket):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            pred = self._model_call(bucket)
            pred_real = self.mad * pred + self.mean
            target = ((bucket.label - self.mean) / self.mad)
            loss = self.loss_fn(pred, target)
        return loss, pred_real

    def _run_eager(self, nodes, positions, node_mask, edges, dense_edge_mask,
                   label, n_nodes, sparse_edges):
        """Safe overflow path once the bounded graph cache is full."""
        self.optimizer.zero_grad(set_to_none=False)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            pred = self.model(
                h0=nodes, x=positions, edges=edges, edge_attr=None,
                node_mask=node_mask, edge_mask=dense_edge_mask,
                n_nodes=n_nodes, sparse_edges_per_layer=sparse_edges).float()
            pred_real = self.mad * pred + self.mean
            loss = self.loss_fn(pred, (label - self.mean) / self.mad)
        loss.backward()
        self.optimizer.step()
        return loss, pred_real

    def _snapshot_training_state(self):
        params = [(p, p.detach().clone()) for p in self.model.parameters()]
        state = []
        for values in self.optimizer.state.values():
            for value in values.values():
                if torch.is_tensor(value):
                    state.append((value, value.detach().clone()))
        return params, state

    @staticmethod
    @torch.no_grad()
    def _restore_training_state(snapshot):
        params, state = snapshot
        for target, saved in params:
            target.copy_(saved)
        for target, saved in state:
            target.copy_(saved)

    def _allocate(self, key, node_features):
        # One complete, masked graph is appended.  Padded dense and sparse edges
        # land at its first node and can never influence a real prediction.
        bucket = _Bucket()
        bucket.key = key
        device, dtype = node_features.device, node_features.dtype
        total_nodes = (key.batch_size + 1) * key.n_nodes
        scratch = key.batch_size * key.n_nodes
        bucket.scratch_node = scratch

        bucket.nodes = torch.zeros(
            total_nodes, node_features.size(1), device=device, dtype=dtype)
        bucket.positions = torch.zeros(
            total_nodes, 3, device=device, dtype=dtype)
        bucket.node_mask = torch.zeros(
            total_nodes, 1, device=device, dtype=dtype)
        bucket.label = torch.empty(
            key.batch_size, device=device, dtype=torch.float32)

        bucket.dense_rows = torch.full(
            (key.dense_capacity,), scratch, device=device, dtype=torch.long)
        bucket.dense_cols = torch.full(
            (key.dense_capacity,), scratch, device=device, dtype=torch.long)
        bucket.dense_mask = torch.zeros(
            key.dense_capacity, 1, device=device, dtype=dtype)

        # Fused dense layers memoize row segment boundaries.  A graph bucket
        # intentionally reuses one row tensor for different batches, so keep
        # that cached output at a stable address and refresh its values before
        # every replay.  Without this, repeated bucket hits use stale segments.
        from EGNN.fused_dense import row_offsets
        bucket.node_ids = torch.arange(
            total_nodes + 1, device=device, dtype=torch.long)
        bucket.dense_offsets = row_offsets(bucket.dense_rows, total_nodes)

        sparse_edges = []
        cap_iter = iter(key.sparse_capacities)
        for layer in range(self.n_sparse_layers):
            if layer < self.sparse_start:
                row = torch.empty(0, device=device, dtype=torch.long)
                col = torch.empty(0, device=device, dtype=torch.long)
                mask = torch.empty((0, 1), device=device, dtype=dtype)
            else:
                cap = next(cap_iter)
                row = torch.full((cap,), scratch, device=device, dtype=torch.long)
                col = torch.full((cap,), scratch, device=device, dtype=torch.long)
                mask = torch.zeros((cap, 1), device=device, dtype=dtype)
            sparse_edges.append((row, col, mask))
        bucket.sparse_edges = sparse_edges if self.n_sparse_layers else None
        bucket.graph = torch.cuda.CUDAGraph()
        return bucket

    @torch.no_grad()
    def _copy_inputs(self, bucket, nodes, positions, node_mask, edges,
                     dense_edge_mask, label, sparse_edges):
        real_nodes = bucket.key.batch_size * bucket.key.n_nodes
        bucket.nodes[:real_nodes].copy_(nodes)
        bucket.positions[:real_nodes].copy_(positions)
        bucket.node_mask[:real_nodes].copy_(node_mask)
        bucket.label.copy_(label)

        dense_count = edges[0].numel()
        bucket.dense_rows.fill_(bucket.scratch_node)
        bucket.dense_cols.fill_(bucket.scratch_node)
        bucket.dense_rows[:dense_count].copy_(edges[0])
        bucket.dense_cols[:dense_count].copy_(edges[1])
        bucket.dense_mask.zero_()
        if dense_edge_mask is None:
            bucket.dense_mask[:dense_count].fill_(1)
        else:
            bucket.dense_mask[:dense_count].copy_(dense_edge_mask)
        torch.searchsorted(bucket.dense_rows, bucket.node_ids,
                           out=bucket.dense_offsets)

        if self.n_sparse_layers:
            for layer in range(self.sparse_start, self.n_sparse_layers):
                src_row, src_col, _ = sparse_edges[layer]
                dst_row, dst_col, dst_mask = bucket.sparse_edges[layer]
                count = src_row.numel()
                dst_row.fill_(bucket.scratch_node)
                dst_col.fill_(bucket.scratch_node)
                dst_row[:count].copy_(src_row)
                dst_col[:count].copy_(src_col)
                dst_mask.zero_()
                dst_mask[:count].fill_(1)

    def _warmup_train(self, bucket, iterations=2):
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
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.zero_()
        torch.cuda.synchronize()

    def _capture(self, bucket):
        self._warmup_train(bucket)

        # Capture executes the step once.  Restore it afterwards so the caller's
        # first replay performs exactly one update, just like every cache hit.
        snapshot = self._snapshot_training_state()
        with torch.cuda.graph(bucket.graph, pool=self.pool):
            self.optimizer.zero_grad(set_to_none=False)
            bucket.loss, bucket.pred_real = self._forward_loss(bucket)
            bucket.loss.backward()
            self.optimizer.step()
        self._restore_training_state(snapshot)
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.zero_()
        torch.cuda.synchronize()

        self.captures += 1
        print("CUDA graph captured: "
              f"bs={bucket.key.batch_size} n={bucket.key.n_nodes} "
              f"dense={bucket.key.dense_capacity} "
              f"sparse={bucket.key.sparse_capacities}")

    def run(self, nodes, positions, node_mask, edges, dense_edge_mask, label,
            n_nodes,
            sparse_edges: Optional[Sequence] = None):
        batch_size = label.numel()
        key = self._key(batch_size, n_nodes, edges, sparse_edges)
        bucket = self.buckets.get(key)
        if bucket is None:
            # CUDA/cuBLAS graph state becomes unreliable when too many matrix
            # shapes are retained.  Less common shapes execute eagerly once the
            # bounded cache is full instead of risking capture invalidation.
            if len(self.buckets) >= self.max_buckets:
                self.eager_fallbacks += 1
                if key not in self.fallback_keys:
                    self.fallback_keys.add(key)
                    print("CUDA graph cache full; using eager fallback for "
                          f"bs={key.batch_size} n={key.n_nodes} "
                          f"dense={key.dense_capacity}")
                return self._run_eager(
                    nodes, positions, node_mask, edges, dense_edge_mask,
                    label, n_nodes, sparse_edges)
            bucket = self._allocate(key, nodes)
            self.buckets[key] = bucket
            self._copy_inputs(bucket, nodes, positions, node_mask, edges,
                              dense_edge_mask, label, sparse_edges)
            self._capture(bucket)
        else:
            self._copy_inputs(bucket, nodes, positions, node_mask, edges,
                              dense_edge_mask, label, sparse_edges)

        bucket.graph.replay()
        return bucket.loss, bucket.pred_real

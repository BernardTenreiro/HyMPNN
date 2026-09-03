from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Embedding, Linear
from torch_geometric.nn import radius_graph
from torch_geometric.utils import scatter

# Atomic Mass Table for frame coloring
_ATOMIC_MASS: Dict[int, float] = {
    1: 1.008,
    2: 4.003,
    3: 6.941,
    4: 9.012,
    5: 10.811,
    6: 12.011,
    7: 14.007,
    8: 15.999,
    9: 18.998,
    10: 20.180,
    11: 22.990,
    12: 24.305,
    13: 26.982,
    14: 28.086,
    15: 30.974,
    16: 32.065,
    17: 35.453,
    18: 39.948,
    19: 39.098,
    20: 40.078,
    35: 79.904,
    53: 126.904,
}
_MAX_Z = 100


def _build_mass_table(max_z: int = _MAX_Z) -> Tensor:
    mass_table = torch.zeros(max_z + 1, dtype=torch.float32)
    for z, mass in _ATOMIC_MASS.items():
        if z <= max_z:
            mass_table[z] = mass
    for z in range(max_z + 1):
        if mass_table[z] == 0:
            mass_table[z] = float(z)
    return mass_table


# Building blocks
class ResidualLayer(nn.Module):
    def __init__(self, hidden_channels: int, act: Callable):
        super().__init__()
        self.act = act
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.act(self.lin2(self.act(self.lin1(x))))


class EGNNMessageBlock(nn.Module):
    def __init__(
        self, hidden_channels: int, edge_emb_dim: int, act: Callable, num_residual: int = 2
    ):
        super().__init__()
        self.act = act
        self.msg_mlp = nn.Sequential(
            Linear(2 * hidden_channels + edge_emb_dim, hidden_channels),
            act,
            Linear(hidden_channels, hidden_channels),
        )
        self.upd_mlp = nn.Sequential(
            Linear(2 * hidden_channels, hidden_channels),
            act,
            Linear(hidden_channels, hidden_channels),
        )
        self.residuals = nn.ModuleList(
            [ResidualLayer(hidden_channels, act) for _ in range(num_residual)]
        )

    def forward(
        self,
        x: Tensor,
        edge_emb: Tensor,
        edge_index: Tensor,
        edge_mask: Tensor,
        num_nodes: int,
    ) -> Tensor:
        if edge_mask.numel() == 0 or edge_mask.sum().item() == 0:
            return x

        sel = edge_mask.bool()
        dst = edge_index[0, sel]
        src = edge_index[1, sel]
        e_sel = edge_emb[sel]

        msg = self.msg_mlp(torch.cat([x[dst], x[src], e_sel], dim=-1))
        agg = scatter(msg, dst, dim=0, dim_size=num_nodes, reduce="sum")
        x = x + self.act(self.upd_mlp(torch.cat([x, agg], dim=-1)))
        for res in self.residuals:
            x = res(x)
        return x


# Readout MLP: Node features -> per graph scalar
class ReadoutMLP(nn.Module):
    """Simple MLP that maps node embeddings to per-node scalars, then sums over the graph."""

    def __init__(self, hidden_channels: int, out_channels: int, act: Callable, num_layers: int = 2):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = hidden_channels
        for _ in range(num_layers):
            layers.append(Linear(in_dim, hidden_channels))
            layers.append(act)
            in_dim = hidden_channels
        layers.append(Linear(in_dim, out_channels))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        per_node = self.net(x)  # (N, out_channels)
        out = scatter(per_node, batch, dim=0, reduce="sum")  # (B, out_channels)
        return out


# Activation helper
def _resolve_activation(act: Union[str, Callable]) -> Callable:
    if callable(act):
        return act
    table: Dict[str, Callable] = {
        "swish": nn.SiLU(),
        "silu": nn.SiLU(),
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "identity": nn.Identity(),
    }
    key = act.lower()
    if key not in table:
        raise ValueError(f"Unsupported activation '{act}'.")
    return table[key]


# Frame Ordering/ Edge Coloring
def node_scores_from_mode(z: Tensor, scoring: str, mass_table: Tensor) -> Tensor:
    if scoring == "atomic_number":
        return z.float()
    mass = mass_table[z]
    if scoring == "mass":
        return mass
    if scoring == "mass_noh":
        mass = mass.clone()
        mass[z == 1] = 0.0
        return mass
    if scoring == "penalized_h":
        mass = mass.clone()
        mass[z == 1] = -1.0
        return mass
    raise ValueError(f"Unknown frame_scoring='{scoring}'.")


def unique_undirected_pairs(edge_index: Tensor) -> Tuple[List[Tuple[int, int]], Tensor]:
    e_count = edge_index.size(1)
    pair_map: Dict[Tuple[int, int], int] = {}
    pairs: List[Tuple[int, int]] = []
    edge_to_pair = torch.empty(e_count, dtype=torch.long, device=edge_index.device)

    for e in range(e_count):
        a = int(edge_index[0, e])
        b = int(edge_index[1, e])
        u, v = (a, b) if a < b else (b, a)
        key = (u, v)
        idx = pair_map.get(key)
        if idx is None:
            idx = len(pairs)
            pair_map[key] = idx
            pairs.append(key)
        edge_to_pair[e] = idx

    return pairs, edge_to_pair


def greedy_pair_coloring(pairs: Sequence[Tuple[int, int]]) -> List[int]:
    colors = [-1] * len(pairs)
    node_to_pairs: Dict[int, List[int]] = {}

    for p_idx, (u, v) in enumerate(pairs):
        node_to_pairs.setdefault(u, []).append(p_idx)
        node_to_pairs.setdefault(v, []).append(p_idx)

    for p_idx, (u, v) in enumerate(pairs):
        used = {colors[nbr] for nbr in node_to_pairs[u] + node_to_pairs[v] if colors[nbr] != -1}
        c = 0
        while c in used:
            c += 1
        colors[p_idx] = c

    return colors


def directed_color_masks_from_undirected(edge_index: Tensor) -> List[Tensor]:
    pairs, edge_to_pair = unique_undirected_pairs(edge_index)
    if len(pairs) == 0:
        return []
    pair_colors = greedy_pair_coloring(pairs)
    num_colors = max(pair_colors) + 1
    pair_colors_t = torch.tensor(pair_colors, dtype=torch.long, device=edge_index.device)
    edge_colors = pair_colors_t[edge_to_pair]
    return [(edge_colors == c) for c in range(num_colors)]


def score_color_classes(
    edge_coloring: Sequence[Tensor],
    edge_index: Tensor,
    node_scores: Tensor,
) -> List[float]:
    scores: List[float] = []
    for mask in edge_coloring:
        mask = mask.to(edge_index.device).bool()
        if mask.numel() == 0 or mask.sum().item() == 0:
            scores.append(float("inf"))
            continue
        dst = edge_index[0, mask]
        src = edge_index[1, mask]
        scores.append((node_scores[dst] + node_scores[src]).mean().item())
    return scores


def sandwich_order(sorted_colors: Sequence[int]) -> List[int]:
    out: List[int] = []
    lo, hi = 0, len(sorted_colors) - 1
    take_low = True
    while lo <= hi:
        if take_low:
            out.append(int(sorted_colors[lo]))
            lo += 1
        else:
            out.append(int(sorted_colors[hi]))
            hi -= 1
        take_low = not take_low
    return out


def build_frame_schedule(
    edge_coloring: Sequence[Tensor],
    edge_index: Tensor,
    z: Tensor,
    num_blocks: int,
    frame_ordering: str,
    frame_scoring: str,
    mass_table: Tensor,
) -> List[int]:
    scoring_override = {
        "sandwich_atomic": "atomic_number",
        "sandwich_mass": "mass",
        "sandwich_mass_noh": "mass_noh",
        "sandwich_penalized_h": "penalized_h",
    }
    scoring = scoring_override.get(frame_ordering, frame_scoring)
    node_scores = node_scores_from_mode(z, scoring, mass_table)
    scores = score_color_classes(edge_coloring, edge_index, node_scores)
    sorted_colors = sorted(range(len(scores)), key=lambda c: (scores[c], c))

    if not sorted_colors:
        raise ValueError("No color classes available to build a frame schedule.")

    if frame_ordering == "sort_repeat":
        base = sorted_colors
    elif frame_ordering in scoring_override:
        base = sandwich_order(sorted_colors)
    else:
        raise ValueError(
            f"Unsupported frame_ordering='{frame_ordering}'. "
            "Supported: 'sort_repeat', 'sandwich_atomic', 'sandwich_mass', "
            "'sandwich_mass_noh', 'sandwich_penalized_h'."
        )

    return [base[t % len(base)] for t in range(num_blocks)]


# Main model: EGnn with frame ordering and coloring
class SparseEGNN(nn.Module):
    """
    Sparse EGNN with frame-ordered edge coloring.

    Architecture
    ------------
    - Atom embedding  z  →  h  (Embedding)
    - Edge embedding  [h_i || h_j || rbf(d_ij)]  →  e_ij  (MLP)
      where rbf is a simple Gaussian / distance expansion (no DimeNet envelope)
    - num_blocks × EGNNMessageBlock, each operating on one color class
      determined by build_frame_schedule
    - Edge embeddings are residually updated after every block
    - Readout: MLP on final node features h, summed per graph

    No DimeNet components (no Envelope, BesselBasis, OutputPPBlock, glorot_orthogonal).
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        out_channels: int = 1,
        num_blocks: int = 8,
        num_rbf: int = 16,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        num_readout_layers: int = 2,
        num_residual: int = 2,
        act: Union[str, Callable] = "swish",
        frame_ordering: str = "sort_repeat",
        frame_scoring: str = "atomic_number",
        auto_color: bool = True,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.num_blocks = num_blocks
        self.num_rbf = num_rbf
        self.frame_ordering = frame_ordering
        self.frame_scoring = frame_scoring
        self.auto_color = auto_color

        self.register_buffer("mass_table", _build_mass_table())
        # Learnable RBF centres (Gaussian basis)
        self.register_buffer(
            "rbf_centers",
            torch.linspace(0.0, cutoff, num_rbf),
        )
        self.rbf_width = (cutoff / num_rbf) ** 2

        act_fn = _resolve_activation(act)

        self.atom_emb = Embedding(100, hidden_channels)

        # Edge embedding: node pair + distance basis → hidden
        self.edge_emb_mlp = nn.Sequential(
            Linear(2 * hidden_channels + num_rbf, hidden_channels),
            act_fn,
            Linear(hidden_channels, hidden_channels),
        )

        # Message-passing blocks (one per scheduled step)
        self.mp_blocks = nn.ModuleList(
            [
                EGNNMessageBlock(hidden_channels, hidden_channels, act_fn, num_residual)
                for _ in range(num_blocks)
            ]
        )

        # Final readout MLP: h → scalar per node → sum per graph
        self.readout = ReadoutMLP(hidden_channels, out_channels, act_fn, num_readout_layers)

    # ------------------------------------------------------------------
    # Distance → RBF features (simple Gaussian basis, no DimeNet parts)
    # ------------------------------------------------------------------
    def _rbf(self, dist: Tensor) -> Tensor:
        """Gaussian radial basis: exp(-||d - c||^2 / w)."""
        diff = dist.unsqueeze(-1) - self.rbf_centers  # (E, num_rbf)
        return torch.exp(-diff.pow(2) / self.rbf_width)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        batch: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
        edge_coloring: Optional[Sequence[Tensor]] = None,
        return_schedule: bool = False,
    ):
        if batch is None:
            batch = torch.zeros(z.size(0), dtype=torch.long, device=z.device)

        # Build radius graph if not provided
        if edge_index is None:
            edge_index = radius_graph(
                pos,
                r=self.cutoff,
                batch=batch,
                max_num_neighbors=self.max_num_neighbors,
            )

        # Build edge coloring if not provided
        if edge_coloring is None:
            if not self.auto_color:
                raise ValueError("edge_coloring must be provided when auto_color=False.")
            edge_coloring = directed_color_masks_from_undirected(edge_index)

        num_nodes = z.size(0)
        dst, src = edge_index

        # Pairwise distances → RBF
        dist = (pos[dst] - pos[src]).pow(2).sum(dim=-1).sqrt()  # (E,)
        rbf = self._rbf(dist)  # (E, num_rbf)

        # Initial node and edge embeddings
        h = self.atom_emb(z)  # (N, C)
        edge_emb = self.edge_emb_mlp(torch.cat([h[dst], h[src], rbf], dim=-1))  # (E, C)

        # Frame schedule over color classes
        schedule = build_frame_schedule(
            edge_coloring=edge_coloring,
            edge_index=edge_index,
            z=z,
            num_blocks=self.num_blocks,
            frame_ordering=self.frame_ordering,
            frame_scoring=self.frame_scoring,
            mass_table=self.mass_table,
        )

        # Message passing
        for t, color_idx in enumerate(schedule):
            mask = edge_coloring[color_idx].to(h.device).bool()
            h = self.mp_blocks[t](h, edge_emb, edge_index, mask, num_nodes)
            # Residually update edge embeddings with refreshed node features
            edge_emb = edge_emb + self.edge_emb_mlp(torch.cat([h[dst], h[src], rbf], dim=-1))

        # Readout: MLP on node features, sum-pooled per graph
        out = self.readout(h, batch)  # (B, out_channels)

        if return_schedule:
            return out, schedule
        return out

"""Dense, sparse, pairwise, and hybrid EGNN model definitions."""

import torch
from torch import nn

from ..data.qm9.edge_scheduling import build_atomic_mass_table
from .layers import EquivariantGraphConvolution


class MaskedEquivariantGraphConvolution(EquivariantGraphConvolution):
    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        edges_in_d=0,
        nodes_attr_dim=0,
        act_fn=nn.ReLU(),
        recurrent=True,
        coords_weight=1.0,
        attention=False,
    ):
        super().__init__(
            input_features=input_nf,
            output_features=output_nf,
            hidden_features=hidden_nf,
            edge_features=edges_in_d,
            node_attribute_features=nodes_attr_dim,
            activation=act_fn,
            recurrent=recurrent,
            coordinate_weight=coords_weight,
            attention=attention,
        )
        del self.coord_mlp

    def forward(
        self,
        h,
        edge_index,
        coord,
        node_mask,
        edge_mask,
        edge_attr=None,
        node_attr=None,
        n_nodes=None,
    ):
        row, col = edge_index
        radial, _ = self.coordinate_features(edge_index, coord)
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        edge_feat = edge_feat * edge_mask
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        return h, coord, edge_attr


class EGNN(nn.Module):
    """Standard EGNN"""

    def __init__(
        self,
        in_node_nf,
        in_edge_nf,
        hidden_nf,
        device="cpu",
        act_fn=nn.SiLU(),
        n_layers=4,
        coords_weight=1.0,
        attention=False,
        node_attr=1,
    ):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        self.node_attr = node_attr
        n_node_attr = in_node_nf if node_attr else 0
        for i in range(n_layers):
            self.add_module(
                "gcl_%d" % i,
                MaskedEquivariantGraphConvolution(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    edges_in_d=in_edge_nf,
                    nodes_attr_dim=n_node_attr,
                    act_fn=act_fn,
                    recurrent=True,
                    coords_weight=coords_weight,
                    attention=attention,
                ),
            )
        self.node_dec = nn.Sequential(
            nn.Linear(self.hidden_nf, self.hidden_nf),
            act_fn,
            nn.Linear(self.hidden_nf, self.hidden_nf),
        )
        self.graph_dec = nn.Sequential(
            nn.Linear(self.hidden_nf, self.hidden_nf), act_fn, nn.Linear(self.hidden_nf, 1)
        )
        self.to(self.device)

    def forward(
        self,
        h0,
        x,
        edges,
        edge_attr,
        node_mask,
        edge_mask,
        n_nodes,
        charges=None,
        sparse_edges_per_layer=None,
    ):
        h = self.embedding(h0)
        for i in range(self.n_layers):
            if self.node_attr:
                h, _, _ = self._modules["gcl_%d" % i](
                    h,
                    edges,
                    x,
                    node_mask,
                    edge_mask,
                    edge_attr=edge_attr,
                    node_attr=h0,
                    n_nodes=n_nodes,
                )
            else:
                h, _, _ = self._modules["gcl_%d" % i](
                    h,
                    edges,
                    x,
                    node_mask,
                    edge_mask,
                    edge_attr=edge_attr,
                    node_attr=None,
                    n_nodes=n_nodes,
                )
        h = self.node_dec(h)
        h = h * node_mask
        h = h.view(-1, n_nodes, self.hidden_nf)
        h = torch.sum(h, dim=1)
        pred = self.graph_dec(h)
        return pred.squeeze(1)


###############################################################################
# Pairwise equivariant graph convolution
###############################################################################


class PairwiseEquivariantGraphConvolution(nn.Module):
    """
    Sparse/pairwise version of E_GCL.

    Performs the same message-passing operations as E_GCL, but only on
    the selected node pairs given by `rows` and `cols`.

    For every selected pair (i, j), both directed edges are evaluated:

        i -> j
        j -> i

    Thus the pairwise layer preserves the same directional message-passing
    structure as the standard EGNN.

    Coordinates are not updated, matching E_GCL_mask.
    """

    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        edges_in_d=0,
        nodes_attr_dim=0,
        act_fn=nn.SiLU(),
        recurrent=True,
        attention=False,
    ):
        super().__init__()

        input_edge = input_nf * 2

        self.hidden_nf = hidden_nf
        self.recurrent = recurrent
        self.attention = attention

        # Same edge MLP as E_GCL
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + 1 + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )

        # Same node MLP as E_GCL
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_attr_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )

        # Same attention mechanism as E_GCL
        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr=None):
        if edge_attr is None:
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)

        out = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val

        return out

    def forward(self, h, x, rows, cols, node_attr=None, edge_attr=None):
        """
        Args:
            h:
                (total_nodes, hidden_nf)

            x:
                (total_nodes, 3)

            rows:
                (num_pairs,) first node of each selected pair

            cols:
                (num_pairs,) second node of each selected pair

            node_attr:
                (total_nodes, nodes_attr_dim)

            edge_attr:
                Optional edge attributes for selected pairs.

        Returns:
            h_updated:
                Updated node features.

        Notes:
            rows/cols are assumed to form a matching, so every node
            participates in at most one selected pair.
        """

        if rows.size(0) == 0:
            return h

        # ============================================================
        # Coordinates / radial
        # ============================================================

        coord_diff = x[rows] - x[cols]

        radial = torch.sum(coord_diff**2, dim=1, keepdim=True)

        # ============================================================
        # Forward direction: i -> j
        #
        # Same as:
        # edge_model(h[row], h[col], radial)
        # ============================================================

        h_i = h[rows]
        h_j = h[cols]

        edge_ij = self.edge_model(h_i, h_j, radial, edge_attr)

        # ============================================================
        # Reverse direction: j -> i
        #
        # Same as:
        # edge_model(h[col], h[row], radial)
        # ============================================================

        edge_ji = self.edge_model(h_j, h_i, radial, edge_attr)

        # ============================================================
        # Node updates
        #
        # In E_GCL:
        #
        # row, col = edge_index
        # agg = unsorted_segment_sum(edge_attr, row, ...)
        #
        # Here:
        #
        # i receives edge_ji
        # j receives edge_ij
        #
        # because messages are aggregated according to the destination.
        # ============================================================

        if node_attr is not None:
            node_attr_i = node_attr[rows]
            node_attr_j = node_attr[cols]
        else:
            node_attr_i = None
            node_attr_j = None

        # Node i receives message from j -> i
        if node_attr_i is not None:
            agg_i = torch.cat([h_i, edge_ji, node_attr_i], dim=1)
        else:
            agg_i = torch.cat([h_i, edge_ji], dim=1)

        # Node j receives message from i -> j
        if node_attr_j is not None:
            agg_j = torch.cat([h_j, edge_ij, node_attr_j], dim=1)
        else:
            agg_j = torch.cat([h_j, edge_ij], dim=1)

        # ============================================================
        # Node MLP
        # ============================================================

        h_i_new = self.node_mlp(agg_i)
        h_j_new = self.node_mlp(agg_j)

        # Same recurrent update as E_GCL
        if self.recurrent:
            h_i_new = h_i + h_i_new
            h_j_new = h_j + h_j_new

        # ============================================================
        # Write updated nodes back
        # ============================================================

        h_updated = h.clone()

        h_updated[rows] = h_i_new
        h_updated[cols] = h_j_new

        return h_updated


###############################################################################
# Pairwise Joint Update Layer
###############################################################################


class JointPairwiseLayer(nn.Module):
    """
    Pairwise update with a single learnable function.

    Unlike the standard EGNN, there is no separate message function
    followed by aggregation and then a node-update function.

    For each selected pair (i, j):

        h_i' = h_i + f(h_i, h_j, radial)
        h_j' = h_j + f(h_j, h_i, radial)

    The same learnable function f is used for both directions.

    Because the selected edges form a matching, there is no need for
    message aggregation: each node interacts with at most one other
    node in this layer.

    Nodes that are not part of a selected pair remain unchanged.

    Coordinates are not updated, matching E_GCL_mask.
    """

    def __init__(self, hidden_nf, act_fn=nn.SiLU(), node_attr_dim=0):
        super().__init__()

        self.hidden_nf = hidden_nf
        self.node_attr_dim = node_attr_dim

        # Input:
        #
        #   h_i       -> hidden_nf
        #   h_j       -> hidden_nf
        #   radial    -> 1
        #   node_attr -> node_attr_dim (optional)
        #
        # Output:
        #
        #   update -> hidden_nf

        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_nf + 1 + node_attr_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
        )

    def forward(self, h, x, rows, cols, node_attr=None):
        """
        Args:
            h:
                (total_nodes, hidden_nf)

            x:
                (total_nodes, 3)

            rows:
                (num_pairs,) first endpoint of each pair

            cols:
                (num_pairs,) second endpoint of each pair

            node_attr:
                (total_nodes, node_attr_dim), optional

        Returns:
            h_updated:
                (total_nodes, hidden_nf)
        """

        if rows.size(0) == 0:
            return h

        # ============================================================
        # Select pair endpoints
        # ============================================================

        h_i = h[rows]
        h_j = h[cols]

        # ============================================================
        # Radial distance
        # ============================================================

        coord_diff = x[rows] - x[cols]

        radial = torch.sum(coord_diff**2, dim=1, keepdim=True)

        # ============================================================
        # Node attributes
        # ============================================================

        if node_attr is not None:
            node_attr_i = node_attr[rows]
            node_attr_j = node_attr[cols]
        else:
            node_attr_i = None
            node_attr_j = None

        # ============================================================
        # Direct pairwise update
        #
        # h_i' = h_i + f(h_i, h_j, radial)
        # h_j' = h_j + f(h_j, h_i, radial)
        #
        # Same learnable function is used in both directions.
        # ============================================================

        if node_attr_i is not None:
            input_i = torch.cat([h_i, h_j, radial, node_attr_i], dim=-1)

            input_j = torch.cat([h_j, h_i, radial, node_attr_j], dim=-1)

        else:
            input_i = torch.cat([h_i, h_j, radial], dim=-1)

            input_j = torch.cat([h_j, h_i, radial], dim=-1)

        # ============================================================
        # One batched neural-network call
        # ============================================================

        update_input = torch.cat([input_i, input_j], dim=0)

        updates = self.update_mlp(update_input)

        update_i, update_j = torch.chunk(updates, 2, dim=0)

        # ============================================================
        # Residual updates
        # ============================================================

        h_i_new = h_i + update_i
        h_j_new = h_j + update_j

        # ============================================================
        # Write back into full node tensor
        # ============================================================

        h_updated = h.clone()

        h_updated[rows] = h_i_new
        h_updated[cols] = h_j_new

        return h_updated


###############################################################################
# Symmetric pairwise layer
###############################################################################


class SymmetricPairwiseLayer(nn.Module):
    """
    Pairwise update using a single symmetric MLP.

    The pair representation is:
        h_s = h_i + h_j
        h_d = h_i - h_j

    The MLP receives only symmetric quantities:
        [h_s, |h_d|, radial]

    The MLP produces a single symmetric update:
        z = f(h_s, |h_d|, radial)

    Both endpoints receive the same update:
        h_i_new = h_i + z
        h_j_new = h_j + z

    This is permutation equivariant because swapping i <-> j leaves
    h_s, |h_d|, and radial unchanged, while the same update is applied
    to both endpoints.

    Nodes NOT in the matching pass through unchanged.
    """

    def __init__(self, hidden_nf, act_fn=nn.SiLU()):
        super().__init__()

        self.hidden_nf = hidden_nf

        # Input:
        #   h_s       -> hidden_nf
        #   |h_d|     -> hidden_nf
        #   radial     -> 1
        #
        # Total = 2 * hidden_nf + 1

        self.f = nn.Sequential(
            nn.Linear(2 * hidden_nf + 1, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
        )

    def forward(self, h, x, rows, cols):
        """
        Args:
            h:
                (total_nodes, hidden_nf) node embeddings

            x:
                (total_nodes, 3) coordinates

            rows:
                (num_matched_edges,) source indices

            cols:
                (num_matched_edges,) target indices

        Returns:
            h_updated:
                (total_nodes, hidden_nf) updated node embeddings
        """

        if rows.size(0) == 0:
            return h

        h_i = h[rows]
        h_j = h[cols]

        # Coordinate information
        coord_diff = x[rows] - x[cols]

        radial = (coord_diff**2).sum(dim=-1, keepdim=True)

        # Symmetric pair representation
        h_s = h_i + h_j
        h_d = h_i - h_j

        # Single NN call
        z = self.f(torch.cat([h_s, h_d.abs(), radial], dim=-1))

        # Same symmetric update to both endpoints
        h_i_new = h_i + z
        h_j_new = h_j + z

        # Write updates back
        h_updated = h.clone()

        h_updated[rows] = h_i_new
        h_updated[cols] = h_j_new

        return h_updated


###############################################################################
# Pairwise symmetric/asymmetric update layer
###############################################################################

# !!!It is possible to use less NN calls for more speedup. Do not need message and update NNs, since no aggregation.!!!
# !!!Need to edit pairwise nn, it uses the same dimension for standard and pairwise.!!!


class SymmetricAsymmetricPairwiseLayer(nn.Module):
    """
    Both endpoints update simultaneously from the same pre-update state:
        h_s = h_i + h_j                    (symmetric)
        h_d = h_i - h_j                    (antisymmetric)
        z_s = f(h_s, |h_d|, radial)      (symmetric update)
        z_d = h_d * g(|h_d|, radial)        (antisymmetric, odd function)
        h_i_new = h_i + z_s + z_d
        h_j_new = h_j + z_s - z_d

    Swapping i <-> j gives identical equations (permutation equivariant).
    Nodes NOT in the matching pass through unchanged.
    """

    def __init__(self, hidden_nf, act_fn=nn.SiLU()):
        super().__init__()
        self.hidden_nf = hidden_nf

        # f_s: symmetric MLP. Input: h_s + |h_d| + radial = 2*hidden_nf + 1
        self.f_s = nn.Sequential(
            nn.Linear(2 * hidden_nf + 1, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
        )

        # f_d gate: ensures antisymmetry via z_d = h_d * g(|h_d|, radial)
        self.f_d_gate = nn.Sequential(
            nn.Linear(hidden_nf + 1, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            nn.Sigmoid(),
        )

    def forward(self, h, x, rows, cols):
        """
        Args:
            h:    (total_nodes, hidden_nf) node embeddings
            x:    (total_nodes, 3) coordinates
            rows: (num_matched_edges,) source indices
            cols: (num_matched_edges,) target indices
        Returns:
            h_updated: (total_nodes, hidden_nf)
        """
        if rows.size(0) == 0:
            return h

        h_i = h[rows]
        h_j = h[cols]

        coord_diff = x[rows] - x[cols]
        radial = (coord_diff**2).sum(dim=-1, keepdim=True)

        h_s = h_i + h_j
        h_d = h_i - h_j

        h_d_abs = h_d.abs()  # shared by f_s and f_d_gate; was computed twice

        z_s = self.f_s(torch.cat([h_s, h_d_abs, radial], dim=-1))
        gate = self.f_d_gate(torch.cat([h_d_abs, radial], dim=-1))
        z_d = h_d * gate

        h_i_new = h_i + z_s + z_d
        h_j_new = h_j + z_s - z_d

        h_updated = h.clone()
        h_updated[rows] = h_i_new
        h_updated[cols] = h_j_new

        return h_updated


###############################################################################
# PairwiseEGNN
###############################################################################


class PairwiseEGNN(nn.Module):
    """
    EGNN variant using pairwise joint updates instead of standard MP.
    Uses the same coloring/scheduling infrastructure with symmetric/asymmetric updates.
    """

    def __init__(
        self,
        in_node_nf,
        in_edge_nf,
        hidden_nf,
        device="cpu",
        act_fn=nn.SiLU(),
        n_layers=4,
        coords_weight=1.0,
        attention=False,
        node_attr=1,
        frame_ordering="sort_repeat",
        frame_scoring="atomic_number",
    ):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.node_attr = node_attr
        self.frame_ordering = frame_ordering
        self.frame_scoring = frame_scoring

        self.register_buffer("mass_table", build_atomic_mass_table())

        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        for i in range(n_layers):
            self.add_module(f"pairwise_{i}", SymmetricAsymmetricPairwiseLayer(hidden_nf, act_fn))

        self.node_dec = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf), act_fn, nn.Linear(hidden_nf, hidden_nf)
        )
        self.graph_dec = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf), act_fn, nn.Linear(hidden_nf, 1)
        )
        self.to(device)

    def forward(
        self,
        h0,
        x,
        edges,
        edge_attr,
        node_mask,
        edge_mask,
        n_nodes,
        charges=None,
        sparse_edges_per_layer=None,
    ):
        if sparse_edges_per_layer is None:
            raise ValueError("PairwiseEGNN requires sparse_edges_per_layer.")

        h = self.embedding(h0)
        for i in range(self.n_layers):
            sparse_rows, sparse_cols, _ = sparse_edges_per_layer[i]
            h = self._modules[f"pairwise_{i}"](h, x, sparse_rows, sparse_cols)

        h = self.node_dec(h)
        h = h * node_mask
        h = h.view(-1, n_nodes, self.hidden_nf)
        h = torch.sum(h, dim=1)
        pred = self.graph_dec(h)
        return pred.squeeze(1)


###############################################################################
# HybridEGNN (n_standard_layers standard EGNN + n_pairwise_layers pairwise)
###############################################################################
class HybridEGNN(nn.Module):
    """
    Hybrid model:
      - first `n_standard_layers` use standard EGNN message passing
      - `n_pairwise_layers` learnable pairwise modules are applied over
        `n_pairwise_steps` scheduled sparse matchings

    When the number of pairwise steps exceeds the number of pairwise modules,
    modules are reused cyclically. This permits a full edge-color sweep without
    increasing the learned parameter count.

    `sparse_edges_per_layer` must contain `n_standard_layers +
    n_pairwise_steps` entries. The first `n_standard_layers` entries are ignored
    because standard layers use the dense `edges` / `edge_mask` arguments.
    """

    def __init__(
        self,
        in_node_nf,
        in_edge_nf,
        hidden_nf,
        pairwise_nf,
        device="cpu",
        act_fn=nn.SiLU(),
        n_standard_layers=5,
        n_pairwise_layers=3,
        n_pairwise_steps=None,
        coords_weight=1.0,
        attention=False,
        node_attr=1,
        frame_ordering="sort_repeat",
        frame_scoring="atomic_number",
        pairwise_layer_type="sym_asym",
    ):
        super().__init__()

        self.hidden_nf = hidden_nf
        self.pairwise_nf = pairwise_nf
        self.pairwise_layer_type = pairwise_layer_type
        self.device = device
        self.n_standard_layers = n_standard_layers
        self.n_pairwise_layers = n_pairwise_layers
        self.n_pairwise_steps = (
            n_pairwise_layers if n_pairwise_steps is None else n_pairwise_steps
        )
        if self.n_pairwise_steps > 0 and self.n_pairwise_layers == 0:
            raise ValueError("pairwise steps require at least one learnable pairwise layer")
        self.n_layers = n_standard_layers + self.n_pairwise_steps
        self.node_attr = node_attr
        self.frame_ordering = frame_ordering
        self.frame_scoring = frame_scoring
        self.register_buffer("mass_table", build_atomic_mass_table())

        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        n_node_attr = in_node_nf if node_attr else 0

        # Standard EGNN layers first
        for i in range(n_standard_layers):
            self.add_module(
                f"gcl_{i}",
                MaskedEquivariantGraphConvolution(
                    hidden_nf,
                    hidden_nf,
                    hidden_nf,
                    edges_in_d=in_edge_nf,
                    nodes_attr_dim=n_node_attr,
                    act_fn=act_fn,
                    recurrent=True,
                    coords_weight=coords_weight,
                    attention=attention,
                ),
            )

        if hidden_nf != pairwise_nf:
            self.hidden_to_pairwise = nn.Linear(hidden_nf, pairwise_nf)
        else:
            self.hidden_to_pairwise = nn.Identity()

        # Select pairwise layer architecture
        pairwise_layer_types = {
            "sym_asym": SymmetricAsymmetricPairwiseLayer,
            "egcl": PairwiseEquivariantGraphConvolution,
            "symmetric": SymmetricPairwiseLayer,
            "joint": JointPairwiseLayer,
        }

        if pairwise_layer_type not in pairwise_layer_types:
            raise ValueError(
                f"Unknown pairwise_layer_type: {pairwise_layer_type}. "
                f"Choose from {list(pairwise_layer_types.keys())}."
            )

        pairwise_layer_class = pairwise_layer_types[pairwise_layer_type]

        # Pairwise layers after
        for i in range(n_pairwise_layers):
            if pairwise_layer_type == "egcl":
                pairwise_layer = pairwise_layer_class(
                    input_nf=pairwise_nf,
                    output_nf=pairwise_nf,
                    hidden_nf=pairwise_nf,
                    edges_in_d=in_edge_nf,
                    nodes_attr_dim=n_node_attr,
                    act_fn=act_fn,
                    recurrent=True,
                    attention=attention,
                )

            else:
                pairwise_layer = pairwise_layer_class(pairwise_nf, act_fn)

            self.add_module(f"pairwise_{i}", pairwise_layer)

        self.node_dec = nn.Sequential(
            nn.Linear(pairwise_nf, pairwise_nf),
            act_fn,
            nn.Linear(pairwise_nf, pairwise_nf),
        )
        self.graph_dec = nn.Sequential(
            nn.Linear(pairwise_nf, pairwise_nf),
            act_fn,
            nn.Linear(pairwise_nf, 1),
        )

        self.to(self.device)

    def forward(
        self,
        h0,
        x,
        edges,
        edge_attr,
        node_mask,
        edge_mask,
        n_nodes,
        charges=None,
        sparse_edges_per_layer=None,
    ):
        if self.n_pairwise_steps > 0:
            if sparse_edges_per_layer is None:
                raise ValueError("HybridEGNN requires sparse_edges_per_layer for pairwise layers.")

            # The schedule covers both the standard and pairwise layer positions
            # so pairwise layer i reads from index n_standard_layers + i.
            if len(sparse_edges_per_layer) != self.n_layers:
                raise ValueError(
                    f"sparse_edges_per_layer must have exactly {self.n_layers} entries "
                    f"(n_standard_layers={self.n_standard_layers} + "
                    f"n_pairwise_steps={self.n_pairwise_steps}), "
                    f"got {len(sparse_edges_per_layer)}."
                )

        h = self.embedding(h0)

        # Full standard EGNN layers (use dense edges; sparse entries are ignored)
        for i in range(self.n_standard_layers):
            if self.node_attr:
                h, _, _ = self._modules[f"gcl_{i}"](
                    h,
                    edges,
                    x,
                    node_mask,
                    edge_mask,
                    edge_attr=edge_attr,
                    node_attr=h0,
                    n_nodes=n_nodes,
                )
            else:
                h, _, _ = self._modules[f"gcl_{i}"](
                    h,
                    edges,
                    x,
                    node_mask,
                    edge_mask,
                    edge_attr=edge_attr,
                    node_attr=None,
                    n_nodes=n_nodes,
                )

        h = self.hidden_to_pairwise(h)

        # Pairwise schedule entries follow the unused standard-layer entries.
        for step in range(self.n_pairwise_steps):
            layer_index = step % self.n_pairwise_layers
            sparse_rows, sparse_cols, _ = sparse_edges_per_layer[
                self.n_standard_layers + step
            ]

            if self.pairwise_layer_type == "egcl":
                if self.node_attr:
                    h = self._modules[f"pairwise_{layer_index}"](
                        h, x, sparse_rows, sparse_cols, node_attr=h0
                    )

                else:
                    h = self._modules[f"pairwise_{layer_index}"](
                        h, x, sparse_rows, sparse_cols, node_attr=None
                    )

            else:
                h = self._modules[f"pairwise_{layer_index}"](h, x, sparse_rows, sparse_cols)

        h = self.node_dec(h)
        # node_mask must be (total_nodes, 1) to broadcast correctly over hidden_nf
        h = h * node_mask
        h = h.view(-1, n_nodes, self.pairwise_nf)
        h = torch.sum(h, dim=1)
        pred = self.graph_dec(h)
        return pred.squeeze(1)


###############################################################################
# SparseEGNN
###############################################################################


class SparseEGNN(nn.Module):
    def __init__(
        self,
        in_node_nf,
        in_edge_nf,
        hidden_nf,
        device="cpu",
        act_fn=nn.SiLU(),
        n_layers=4,
        coords_weight=1.0,
        attention=False,
        node_attr=1,
        frame_ordering="sort_repeat",
        frame_scoring="atomic_number",
    ):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.node_attr = node_attr
        self.frame_ordering = frame_ordering
        self.frame_scoring = frame_scoring
        self.register_buffer("mass_table", build_atomic_mass_table())
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        n_node_attr = in_node_nf if node_attr else 0
        for i in range(n_layers):
            self.add_module(
                "gcl_%d" % i,
                MaskedEquivariantGraphConvolution(
                    hidden_nf,
                    hidden_nf,
                    hidden_nf,
                    edges_in_d=in_edge_nf,
                    nodes_attr_dim=n_node_attr,
                    act_fn=act_fn,
                    recurrent=True,
                    coords_weight=coords_weight,
                    attention=attention,
                ),
            )
        self.node_dec = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf), act_fn, nn.Linear(hidden_nf, hidden_nf)
        )
        self.graph_dec = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf), act_fn, nn.Linear(hidden_nf, 1)
        )
        self.to(device)

    def forward(
        self,
        h0,
        x,
        edges,
        edge_attr,
        node_mask,
        edge_mask,
        n_nodes,
        charges=None,
        sparse_edges_per_layer=None,
    ):
        if sparse_edges_per_layer is None:
            raise ValueError("SparseEGNN requires sparse_edges_per_layer.")
        h = self.embedding(h0)
        for i in range(self.n_layers):
            sparse_rows, sparse_cols, sparse_emask = sparse_edges_per_layer[i]
            node_attr_i = h0 if self.node_attr else None
            h, _, _ = self._modules["gcl_%d" % i](
                h,
                [sparse_rows, sparse_cols],
                x,
                node_mask,
                sparse_emask,
                edge_attr=None,
                node_attr=node_attr_i,
                n_nodes=n_nodes,
            )
        h = self.node_dec(h)
        h = h * node_mask
        h = h.view(-1, n_nodes, self.hidden_nf)
        h = torch.sum(h, dim=1)
        pred = self.graph_dec(h)
        return pred.squeeze(1)

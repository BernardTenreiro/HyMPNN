from typing import Dict, List, Tuple

import torch
from models.gcl import E_GCL
from torch import nn


class E_GCL_mask(E_GCL):
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
        E_GCL.__init__(
            self,
            input_nf,
            output_nf,
            hidden_nf,
            edges_in_d=edges_in_d,
            nodes_att_dim=nodes_attr_dim,
            act_fn=act_fn,
            recurrent=recurrent,
            coords_weight=coords_weight,
            attention=attention,
        )
        del self.coord_mlp
        self.act_fn = act_fn

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
        radial, coord_diff = self.coord2radial(edge_index, coord)
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
                E_GCL_mask(
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
# Pairwise E_GCL
###############################################################################


class PairwiseEGCL(nn.Module):
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


class PairwiseJointLayer(nn.Module):
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
# PairwiseSymmetricLayer
###############################################################################


class PairwiseSymmetricLayer(nn.Module):
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


class PairwiseSymAsymLayer(nn.Module):
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
    Uses the same coloring/scheduling infrastructure but with PairwiseSymAsymLayer.
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

        self.register_buffer("mass_table", _build_mass_table())

        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        for i in range(n_layers):
            self.add_module(f"pairwise_{i}", PairwiseSymAsymLayer(hidden_nf, act_fn))

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
      - next `n_pairwise_layers` use pairwise updates on scheduled sparse edges

    `sparse_edges_per_layer` must be a list of length `n_standard_layers +
    n_pairwise_layers` (i.e. one entry per total layer).  The first
    `n_standard_layers` entries are ignored by the forward pass (standard layers
    use the dense `edges` / `edge_mask` arguments instead); the remaining
    `n_pairwise_layers` entries are consumed by the pairwise layers in order.

    Callers should build the schedule with:
        n_layers = n_standard_layers + n_pairwise_layers
    so that `assemble_batch_sparse_edges` returns a list of the right length.
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
        self.n_layers = n_standard_layers + n_pairwise_layers
        self.node_attr = node_attr
        self.frame_ordering = frame_ordering
        self.frame_scoring = frame_scoring
        self.register_buffer("mass_table", _build_mass_table())

        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        n_node_attr = in_node_nf if node_attr else 0

        # Standard EGNN layers first
        for i in range(n_standard_layers):
            self.add_module(
                f"gcl_{i}",
                E_GCL_mask(
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
            "sym_asym": PairwiseSymAsymLayer,
            "egcl": PairwiseEGCL,
            "symmetric": PairwiseSymmetricLayer,
            "joint": PairwiseJointLayer,
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
        if self.n_pairwise_layers > 0:
            if sparse_edges_per_layer is None:
                raise ValueError("HybridEGNN requires sparse_edges_per_layer for pairwise layers.")

            # The schedule covers both the standard and pairwise layer positions
            # so pairwise layer i reads from index n_standard_layers + i.
            if len(sparse_edges_per_layer) != self.n_layers:
                raise ValueError(
                    f"sparse_edges_per_layer must have exactly {self.n_layers} entries "
                    f"(n_standard_layers={self.n_standard_layers} + "
                    f"n_pairwise_layers={self.n_pairwise_layers}), "
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
        for i in range(self.n_pairwise_layers):
            sparse_rows, sparse_cols, _ = sparse_edges_per_layer[self.n_standard_layers + i]

            if self.pairwise_layer_type == "egcl":
                if self.node_attr:
                    h = self._modules[f"pairwise_{i}"](h, x, sparse_rows, sparse_cols, node_attr=h0)

                else:
                    h = self._modules[f"pairwise_{i}"](
                        h, x, sparse_rows, sparse_cols, node_attr=None
                    )

            else:
                h = self._modules[f"pairwise_{i}"](h, x, sparse_rows, sparse_cols)

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
        self.register_buffer("mass_table", _build_mass_table())
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        n_node_attr = in_node_nf if node_attr else 0
        for i in range(n_layers):
            self.add_module(
                "gcl_%d" % i,
                E_GCL_mask(
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


###############################################################################
# Edge coloring utilities
###############################################################################

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


def _build_mass_table(max_z: int = _MAX_Z) -> torch.Tensor:
    table = torch.zeros(max_z + 1, dtype=torch.float32)
    for z, m in _ATOMIC_MASS.items():
        if z <= max_z:
            table[z] = m
    for z in range(max_z + 1):
        if table[z] == 0:
            table[z] = float(z)
    return table


def _node_scores(charges, scoring, mass_table):
    """Compute per-node scores for frame ordering."""
    z = charges.long().clamp(0, mass_table.size(0) - 1)
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


def _unique_undirected_pairs(edge_index):
    """Extract unique undirected edges from directed edge_index."""
    rows, cols = edge_index
    e_count = rows.size(0)
    pair_map: Dict[Tuple[int, int], int] = {}
    pairs: List[Tuple[int, int]] = []
    edge_to_pair = torch.empty(e_count, dtype=torch.long)
    for e in range(e_count):
        a, b = int(rows[e]), int(cols[e])
        u, v = (a, b) if a < b else (b, a)
        key = (u, v)
        idx = pair_map.get(key)
        if idx is None:
            idx = len(pairs)
            pair_map[key] = idx
            pairs.append(key)
        edge_to_pair[e] = idx
    return pairs, edge_to_pair


###############################################################################
# Suggestion #1:(Vizing's theorem)
###############################################################################


def _vizing_heuristic_coloring(pairs):
    """
    Improved edge coloring heuristic using networkx line graph + vertex coloring.
    At most D+1 colors used (Vizing's bound) where D is max degree.
    Falls back to greedy if networkx is unavailable.
    """
    try:
        import networkx as nx
        from networkx.algorithms.coloring import greedy_color

        # Build line graph: each undirected edge becomes a node,
        # two nodes are connected if the original edges share an endpoint
        G = nx.Graph()
        for p_idx, (u, v) in enumerate(pairs):
            G.add_edge(u, v, pair_idx=p_idx)

        # Edge coloring = vertex coloring of line graph
        L = nx.line_graph(G)

        # Map line graph nodes back to pair indices
        # Line graph nodes are (u,v) tuples from G's edges
        node_to_pair = {}
        for u, v, data in G.edges(data=True):
            edge_key = (min(u, v), max(u, v))
            node_to_pair[edge_key] = data["pair_idx"]
            # Line graph uses frozenset or tuple as node names
            node_to_pair[(u, v)] = data["pair_idx"]
            node_to_pair[(v, u)] = data["pair_idx"]

        coloring = greedy_color(L, strategy="largest_first")

        colors = [-1] * len(pairs)
        for lg_node, color in coloring.items():
            # lg_node is a frozenset or tuple representing an edge in G
            if isinstance(lg_node, frozenset):
                u, v = sorted(lg_node)
            else:
                u, v = lg_node
            key = (min(u, v), max(u, v))
            if key in node_to_pair:
                colors[node_to_pair[key]] = color

        # Verify all colored
        if -1 in colors:
            # Fallback
            return _greedy_pair_coloring(pairs)
        return colors

    except (ImportError, Exception):
        return _greedy_pair_coloring(pairs)


def _greedy_pair_coloring(pairs):
    """Original greedy coloring as fallback."""
    colors = [-1] * len(pairs)
    node_to_pairs: Dict[int, List[int]] = {}
    for p_idx, (u, v) in enumerate(pairs):
        node_to_pairs.setdefault(u, []).append(p_idx)
        node_to_pairs.setdefault(v, []).append(p_idx)
    for p_idx, (u, v) in enumerate(pairs):
        used = {colors[n] for n in node_to_pairs[u] + node_to_pairs[v] if colors[n] != -1}
        c = 0
        while c in used:
            c += 1
        colors[p_idx] = c
    return colors


def _color_masks_single(edge_index, use_vizing=True):
    """Returns per-color boolean masks over directed edges."""
    pairs, edge_to_pair = _unique_undirected_pairs(edge_index)
    if len(pairs) == 0:
        return []
    if use_vizing:
        pair_colors = _vizing_heuristic_coloring(pairs)
    else:
        pair_colors = _greedy_pair_coloring(pairs)
    num_colors = max(pair_colors) + 1
    pair_colors_t = torch.tensor(pair_colors, dtype=torch.long)
    edge_colors = pair_colors_t[edge_to_pair]
    return [(edge_colors == c) for c in range(num_colors)]


###############################################################################
# Suggestion #2: mass_product scoring
###############################################################################


def _score_color_classes(color_masks, edge_index, node_scores, scoring_method="additive"):
    """
    Score each color class for ordering.

    scoring_method:
        'additive': sum of (score_i + score_j) per edge — original
        'mass_product': sum of (mass_i * mass_j) per edge
    """
    rows, cols = edge_index
    scores = []
    for mask in color_masks:
        m = mask.bool()
        if m.sum().item() == 0:
            scores.append(float("-inf"))
            continue
        if scoring_method == "mass_product":
            scores.append((node_scores[rows[m]] * node_scores[cols[m]]).sum().item())
        else:
            scores.append((node_scores[rows[m]] + node_scores[cols[m]]).mean().item())
    return scores


###############################################################################
# Suggestion #3: K/2 repeat scheduling
###############################################################################


def _sandwich_order(sorted_colors):
    out, lo, hi, take_low = [], 0, len(sorted_colors) - 1, True
    while lo <= hi:
        if take_low:
            out.append(sorted_colors[lo])
            lo += 1
        else:
            out.append(sorted_colors[hi])
            hi -= 1
        take_low = not take_low
    return out


def build_frame_schedule_single(
    edge_index, charges, n_layers, frame_ordering, frame_scoring, mass_table, use_vizing=True
):
    """
    Build coloring + schedule for a single graph.

    frame_ordering options:
        'sort_repeat':     sort colors by score ascending, cycle
        'half_repeat':     use K/2 colors sorted descending, repeat twice
        'sandwich_*':      interleave low/high scored colors

    frame_scoring options:
        'atomic_number', 'mass', 'mass_noh', 'penalized_h': per-node scores
        'mass_product': per-edge product scoring
    """
    scoring_override = {
        "sandwich_atomic": "atomic_number",
        "sandwich_mass": "mass",
        "sandwich_mass_noh": "mass_noh",
        "sandwich_penalized_h": "penalized_h",
    }

    # Determine node scoring
    if frame_scoring == "mass_product":
        node_scoring = "mass"
        color_scoring_method = "mass_product"
    else:
        node_scoring = scoring_override.get(frame_ordering, frame_scoring)
        color_scoring_method = "additive"

    ns = _node_scores(charges, node_scoring, mass_table)
    color_masks = _color_masks_single(edge_index, use_vizing=use_vizing)

    if not color_masks:
        raise ValueError("No edges found — cannot build frame schedule.")

    scores = _score_color_classes(color_masks, edge_index, ns, color_scoring_method)

    if frame_ordering == "sort_repeat":
        sorted_colors = sorted(range(len(scores)), key=lambda c: (scores[c], c))
        base = sorted_colors

    elif frame_ordering == "half_repeat":
        # Suggestion #3: use K/2 unique colors, repeat the sequence twice
        # Sort ascending so lightest pairs come first, heaviest come later
        sorted_colors = sorted(range(len(scores)), key=lambda c: (scores[c], c))
        half_k = max(1, n_layers // 2)
        # Take at most half_k colors (or all if fewer available)
        base_half = sorted_colors[: min(half_k, len(sorted_colors))]
        # Repeat twice
        schedule = (base_half * 2)[:n_layers]
        return color_masks, schedule

    elif frame_ordering in scoring_override:
        sorted_colors = sorted(range(len(scores)), key=lambda c: (scores[c], c))
        base = _sandwich_order(sorted_colors)

    else:
        raise ValueError(f"Unsupported frame_ordering='{frame_ordering}'.")

    schedule = [base[t % len(base)] for t in range(n_layers)]
    return color_masks, schedule


###############################################################################
# Precompute + Assembly
###############################################################################


def precompute_molecule_colorings(
    dataloaders: Dict,
    n_layers: int,
    frame_ordering: str,
    frame_scoring: str,
    use_vizing_coloring: bool = True,
) -> Dict[tuple, Dict[str, List[torch.Tensor]]]:
    """
    Precompute edge colorings for all unique molecules in the dataloaders.

    For HybridEGNN, pass n_layers = n_standard_layers + n_pairwise_layers so
    that the returned cache entries have one schedule slot per total layer.
    The standard-layer slots will simply be unused at inference time.
    """
    mass_table = _build_mass_table()
    cache: Dict[tuple, Dict[str, List[torch.Tensor]]] = {}

    for loader in dataloaders.values():
        for data in loader:
            batch_size, n_nodes, _ = data["positions"].size()
            charges_batch = data["charges"]
            atom_mask_batch = data["atom_mask"]

            for g in range(batch_size):
                n_real = int(atom_mask_batch[g].sum().item())
                if n_real < 2:
                    continue

                charges_g = charges_batch[g, :n_real].long().view(-1)
                cache_key = tuple(charges_g.tolist())
                if cache_key in cache:
                    continue

                idx_i = torch.arange(n_real).unsqueeze(1).expand(n_real, n_real).reshape(-1)
                idx_j = torch.arange(n_real).unsqueeze(0).expand(n_real, n_real).reshape(-1)
                no_self = idx_i != idx_j
                local_rows_t = idx_i[no_self]
                local_cols_t = idx_j[no_self]

                color_masks, schedule = build_frame_schedule_single(
                    edge_index=(local_rows_t, local_cols_t),
                    charges=charges_g,
                    n_layers=n_layers,
                    frame_ordering=frame_ordering,
                    frame_scoring=frame_scoring,
                    mass_table=mass_table,
                    use_vizing=use_vizing_coloring,
                )

                layer_rows, layer_cols, layer_counts = [], [], []
                for layer_idx in range(n_layers):
                    color_idx = schedule[layer_idx]
                    mask = color_masks[color_idx]
                    active_rows = local_rows_t[mask]
                    active_cols = local_cols_t[mask]
                    # One representative per undirected pair. The color masks
                    # are over directed edges and (i,j),(j,i) always share a
                    # color, so every symmetric pairwise update was redundantly
                    # evaluated twice. Keeping row < col preserves both endpoint
                    # updates and their gradients while halving this work.
                    keep = active_rows < active_cols
                    active_rows = active_rows[keep].contiguous()
                    active_cols = active_cols[keep].contiguous()
                    layer_rows.append(active_rows)
                    layer_cols.append(active_cols)
                    layer_counts.append(int(active_rows.numel()))

                cache[cache_key] = {
                    "rows": layer_rows,
                    "cols": layer_cols,
                    "counts": layer_counts,
                }

    print(f"Precomputed colorings for {len(cache)} unique molecules")
    return cache


def assemble_sparse_edges_cpu(
    coloring_cache, charges_batch, atom_mask_batch, n_nodes, n_layers, skip_first=0
):
    """Same edge assembly as :func:`assemble_batch_sparse_edges`, but returns CPU
    tensors and never touches CUDA.

    Split out so the work can run inside a DataLoader worker process (workers
    must not initialise CUDA).  The (rows, cols) produced here are exactly the
    tensors the GPU version would have produced, so results are unchanged.
    """
    if charges_batch.dim() == 3 and charges_batch.size(-1) == 1:
        charges_batch = charges_batch.squeeze(-1)
    batch_size = charges_batch.size(0)

    groups: Dict[tuple, List[int]] = {}
    for g in range(batch_size):
        n_real = int(atom_mask_batch[g].sum().item())
        if n_real < 2:
            continue
        cache_key = tuple(charges_batch[g, :n_real].long().view(-1).tolist())
        if cache_key not in coloring_cache:
            raise RuntimeError(f"Cache miss for charges={cache_key}")
        groups.setdefault(cache_key, []).append(g)

    # HybridEGNN reads slot n_standard_layers + i, so slots below that are never
    # consumed. Building and shipping them was ~half the per-batch edge work.
    per_layer_rows = [[] for _ in range(n_layers)]
    per_layer_cols = [[] for _ in range(n_layers)]
    for cache_key, graph_ids in groups.items():
        cached = coloring_cache[cache_key]
        offsets = torch.tensor(graph_ids, dtype=torch.long) * n_nodes
        for layer_idx in range(skip_first, n_layers):
            base_rows = cached["rows"][layer_idx]
            base_cols = cached["cols"][layer_idx]
            if base_rows.numel() == 0:
                continue
            per_layer_rows[layer_idx].append(
                (base_rows.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)
            )
            per_layer_cols[layer_idx].append(
                (base_cols.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)
            )

    out = []
    for layer_idx in range(n_layers):
        if per_layer_rows[layer_idx]:
            rows = torch.cat(per_layer_rows[layer_idx], dim=0)
            cols = torch.cat(per_layer_cols[layer_idx], dim=0)
        else:
            rows = torch.zeros(0, dtype=torch.long)
            cols = torch.zeros(0, dtype=torch.long)
        out.append((rows, cols))
    return out


class SparseEdgeCollate:
    """collate_fn that also builds the per-layer sparse edge index in the worker.

    Holding the coloring cache on the instance means each forked worker gets it
    copy-on-write, so the per-batch edge assembly (~10 ms of pure Python on the
    training loop's critical path) overlaps with GPU compute instead of blocking it.
    """

    def __init__(self, base_collate, coloring_cache, n_layers, skip_first=0):
        self.base_collate = base_collate
        self.coloring_cache = coloring_cache
        self.n_layers = n_layers
        self.skip_first = skip_first

    def __call__(self, batch):
        out = self.base_collate(batch)
        n_nodes = out["charges"].size(1)
        edges = assemble_sparse_edges_cpu(
            self.coloring_cache,
            out["charges"],
            out["atom_mask"],
            n_nodes,
            self.n_layers,
            self.skip_first,
        )
        out["sparse_rows"] = [r for r, _ in edges]
        out["sparse_cols"] = [c for _, c in edges]
        return out


def assemble_batch_sparse_edges(
    coloring_cache,
    charges_batch,
    atom_mask_batch,
    n_nodes,
    n_layers,
    device,
):
    if charges_batch.dim() == 3 and charges_batch.size(-1) == 1:
        charges_batch = charges_batch.squeeze(-1)
    batch_size = charges_batch.size(0)

    groups: Dict[tuple, List[int]] = {}
    for g in range(batch_size):
        n_real = int(atom_mask_batch[g].sum().item())
        if n_real < 2:
            continue
        charges_g = charges_batch[g, :n_real].long().view(-1)
        cache_key = tuple(charges_g.tolist())
        if cache_key not in coloring_cache:
            raise RuntimeError(f"Cache miss for charges={cache_key}")
        groups.setdefault(cache_key, []).append(g)

    per_layer_rows_cpu = [[] for _ in range(n_layers)]
    per_layer_cols_cpu = [[] for _ in range(n_layers)]

    for cache_key, graph_ids in groups.items():
        cached = coloring_cache[cache_key]
        offsets = torch.tensor(graph_ids, dtype=torch.long) * n_nodes
        for layer_idx in range(n_layers):
            base_rows = cached["rows"][layer_idx]
            base_cols = cached["cols"][layer_idx]
            if base_rows.numel() == 0:
                continue
            rows = (base_rows.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)
            cols = (base_cols.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)
            per_layer_rows_cpu[layer_idx].append(rows)
            per_layer_cols_cpu[layer_idx].append(cols)

    sparse_edges = []
    for layer_idx in range(n_layers):
        if per_layer_rows_cpu[layer_idx]:
            rows = torch.cat(per_layer_rows_cpu[layer_idx], dim=0).to(device, non_blocking=True)
            cols = torch.cat(per_layer_cols_cpu[layer_idx], dim=0).to(device, non_blocking=True)
        else:
            rows = torch.zeros(0, dtype=torch.long, device=device)
            cols = torch.zeros(0, dtype=torch.long, device=device)
        edge_mask = torch.ones(rows.size(0), 1, dtype=torch.float32, device=device)
        sparse_edges.append((rows, cols, edge_mask))
    return sparse_edges

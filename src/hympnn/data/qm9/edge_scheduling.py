"""Edge coloring and per-layer sparse-edge schedule construction."""

from __future__ import annotations

import logging

import torch

_ATOMIC_MASS: dict[int, float] = {
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


def build_atomic_mass_table(max_z: int = _MAX_Z) -> torch.Tensor:
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
    pair_map: dict[tuple[int, int], int] = {}
    pairs: list[tuple[int, int]] = []
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
        graph = nx.Graph()
        for pair_index, (u, v) in enumerate(pairs):
            graph.add_edge(u, v, pair_index=pair_index)

        # Edge coloring = vertex coloring of line graph
        line_graph = nx.line_graph(graph)

        # Map line graph nodes back to pair indices
        # Line graph nodes are (u,v) tuples from G's edges
        node_to_pair = {}
        for u, v, data in graph.edges(data=True):
            edge_key = (min(u, v), max(u, v))
            node_to_pair[edge_key] = data["pair_index"]
            # Line graph uses frozenset or tuple as node names
            node_to_pair[(u, v)] = data["pair_index"]
            node_to_pair[(v, u)] = data["pair_index"]

        coloring = greedy_color(line_graph, strategy="largest_first")

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

    except Exception:
        return _greedy_pair_coloring(pairs)


def _greedy_pair_coloring(pairs):
    """Original greedy coloring as fallback."""
    colors = [-1] * len(pairs)
    node_to_pairs: dict[int, list[int]] = {}
    for pair_index, (u, v) in enumerate(pairs):
        node_to_pairs.setdefault(u, []).append(pair_index)
        node_to_pairs.setdefault(v, []).append(pair_index)
    for pair_index, (u, v) in enumerate(pairs):
        used = {colors[n] for n in node_to_pairs[u] + node_to_pairs[v] if colors[n] != -1}
        color = 0
        while color in used:
            color += 1
        colors[pair_index] = color
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
        active_mask = mask.bool()
        if active_mask.sum().item() == 0:
            scores.append(float("-inf"))
            continue
        if scoring_method == "mass_product":
            scores.append(
                (node_scores[rows[active_mask]] * node_scores[cols[active_mask]]).sum().item()
            )
        else:
            scores.append(
                (node_scores[rows[active_mask]] + node_scores[cols[active_mask]]).mean().item()
            )
    return scores


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

    node_scores = _node_scores(charges, node_scoring, mass_table)
    color_masks = _color_masks_single(edge_index, use_vizing=use_vizing)

    if not color_masks:
        raise ValueError("No edges found — cannot build frame schedule.")

    scores = _score_color_classes(color_masks, edge_index, node_scores, color_scoring_method)

    if frame_ordering == "sort_repeat":
        sorted_colors = sorted(range(len(scores)), key=lambda c: (scores[c], c))
        base = sorted_colors

    elif frame_ordering == "sort_repeat_desc":
        base = sorted(range(len(scores)), key=lambda color: (scores[color], color), reverse=True)

    elif frame_ordering == "half_repeat":
        # Use half of the colors twice, ordered from lightest to heaviest.
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


def precompute_molecule_colorings(
    dataloaders: dict,
    n_layers: int,
    frame_ordering: str,
    frame_scoring: str,
    use_vizing_coloring: bool = True,
) -> dict[tuple, dict[str, list[torch.Tensor]]]:
    """
    Precompute edge colorings for all unique molecules in the dataloaders.

    For HybridEGNN, pass n_layers = n_standard_layers + n_pairwise_layers so
    that the returned cache entries have one schedule slot per total layer.
    The standard-layer slots will simply be unused at inference time.
    """
    mass_table = build_atomic_mass_table()
    cache: dict[tuple, dict[str, list[torch.Tensor]]] = {}

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

    logging.info("Precomputed colorings for %d unique molecules", len(cache))
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

    groups: dict[tuple, list[int]] = {}
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


class SparseEdgeCollator:
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

    groups: dict[tuple, list[int]] = {}
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

from qm9 import dataset
from qm9.models_pairwise import EGNN, PairwiseEGNN, SparseEGNN, HybridEGNN
from qm9.models_pairwise import precompute_molecule_colorings, assemble_batch_sparse_edges
from qm9.models_pairwise import SparseEdgeCollate
from qm9.data.collate import collate_fn as _base_collate
import torch
from torch import nn, optim
import argparse
from qm9 import utils as qm9_utils
import utils
import json
import os
import sys
import time

# Per-run artifacts (architecture.json, losess.json) go next to the code so the
# launch directory does not matter.
EGNN_ROOT = os.path.dirname(os.path.abspath(__file__))

# python -u EGNN/main_qm9_pairwise.py --num_workers 0 --lr 5e-4 --property alpha --exp_name exp_1_alpha --epochs 1 --batch_size 8 --train_fraction 0.01

parser = argparse.ArgumentParser(description='QM9 Example')
parser.add_argument('--exp_name', type=str, default='exp_1')
parser.add_argument('--batch_size', type=int, default=96,
                    help='Batch size. Try larger (256, 512) to see if sparse gets speedup.')
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--log_interval', type=int, default=20)
parser.add_argument('--test_interval', type=int, default=1)
parser.add_argument('--outf', type=str, default=os.path.join(EGNN_ROOT, 'qm9', 'logs'))
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--nf', type=int, default=128)
parser.add_argument('--nf_sparse', type=int, default=None)
parser.add_argument('--attention', type=int, default=1)
parser.add_argument('--n_layers', type=int, default=10,
                    help='Number of layers. For HybridEGNN, total = n_standard_layers + n_pairwise_layers.')
parser.add_argument('--property', type=str, default='homo')
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--charge_power', type=int, default=2)
parser.add_argument('--dataset_paper', type=str, default='cormorant')
parser.add_argument('--node_attr', type=int, default=0)
parser.add_argument('--weight_decay', type=float, default=1e-16)
 
# --- Model selection ---
parser.add_argument(
    '--pairwise_layer_type',
    type=str,
    default='sym_asym',
    choices=['sym_asym', 'egcl', 'symmetric', 'joint'])
parser.add_argument('--pairwise', action='store_true', default=False,
                    help='Use PairwiseEGNN (joint update)')
parser.add_argument('--sparse', action='store_true', default=False,
                    help='Use SparseEGNN (standard MP on sparse edges)')
parser.add_argument('--hybrid', action='store_true', default=False,
                    help='Use HybridEGNN (standard EGNN layers then pairwise layers)')
parser.add_argument('--n_standard_layers', type=int, default=5,
                    help='HybridEGNN only: number of standard EGNN layers.')
parser.add_argument('--n_pairwise_layers', type=int, default=5,
                    help='HybridEGNN only: number of pairwise layers.')
 
# --- Frame scheduling ---
parser.add_argument('--frame_ordering', type=str, default='sort_repeat',
                    choices=['sort_repeat', 'sort_repeat_desc', 'half_repeat',
                             'sandwich_atomic', 'sandwich_mass',
                             'sandwich_mass_noh', 'sandwich_penalized_h'],
                    help='sort_repeat_desc + mass_product for heavy-first. '
                         'half_repeat for K/2 repeat (suggestion #3).')
parser.add_argument('--frame_scoring', type=str, default='atomic_number',
                    choices=['atomic_number', 'mass', 'mass_noh', 'penalized_h',
                             'mass_product'],
                    help='mass_product: score colors by sum(mass_i * mass_j) (suggestion #2)')
 
# --- Coloring ---
parser.add_argument('--amp', action='store_true', default=False,
                    help='bf16 autocast for the forward/backward. Largest remaining GPU lever, '
                         'but a much bigger numerics change (~1e-2 rel) than the fused kernels.')
parser.add_argument('--fused_adam', action='store_true', default=False,
                    help='torch.optim.Adam(fused=True): one kernel instead of ~110 foreach launches. '
                         'Profiled at 1.3 ms CPU issue for 0.4 ms GPU per batch.')
parser.add_argument('--tf32', action='store_true', default=False,
                    help='Allow TF32 tensor-core GEMMs. Changes numerics (~1e-3 rel), unlike the fused kernels.')
parser.add_argument('--fused_dense', action='store_true', default=False,
                    help='Swap E_GCL_mask for the fused CUDA kernels in src/EGNN (helps EGNN and HyEGNN alike).')
parser.add_argument('--fused_pairwise', action='store_true', default=False,
                    help='Swap sym_asym pairwise layers for the fused CUDA kernels in src/HyEGNN.')
parser.add_argument('--cuda_graphs', action='store_true', default=False,
                    help='Hide HybridEGNN launch overhead with size-bucketed CUDA graphs.')
parser.add_argument('--use_vizing_coloring', action='store_true', default=False,
                    help='Use networkx Vizing heuristic edge coloring (tighter than greedy, not guaranteed optimal) (suggestion #1)')
 
# --- Data fraction (suggestion #5) ---
parser.add_argument('--train_fraction', type=float, default=1.0,
                    help='Fraction of training data to use (0.0-1.0). '
                         'Use less data for quick comparison experiments.')
 
args = parser.parse_args()
 
# Validate: at most one model flag
model_flags = [args.pairwise, args.sparse, args.hybrid]
if sum(model_flags) > 1:
    parser.error("Only one of --pairwise, --sparse, --hybrid may be set at a time.")

if args.cuda_graphs and not (
        torch.cuda.is_available() and not args.no_cuda and args.hybrid
        and args.fused_dense and args.fused_pairwise and args.fused_adam
        and args.pairwise_layer_type == 'sym_asym'):
    parser.error("--cuda_graphs requires CUDA, --hybrid, --fused_dense, "
                 "--fused_pairwise, --fused_adam, and sym_asym layers")
 
if args.tf32:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if args.cuda else 'cpu')
dtype = torch.float32
 
utils.makedir(args.outf)
utils.makedir(args.outf + '/' + args.exp_name)
 
dataloaders, charge_scale = dataset.retrieve_dataloaders(args.batch_size, args.num_workers)
 
# --- Suggestion #5: optionally use less training data ---
if args.train_fraction < 1.0:
    train_dataset = dataloaders['train'].dataset
    n_total = len(train_dataset)
    n_use = max(1, int(n_total * args.train_fraction))
    subset_indices = list(range(n_use))
    train_subset = torch.utils.data.Subset(train_dataset, subset_indices)
    dataloaders['train'] = torch.utils.data.DataLoader(
        train_subset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, drop_last=False)
    print(f"Using {n_use}/{n_total} training samples ({args.train_fraction*100:.0f}%)")
 
meann, mad = qm9_utils.compute_mean_mad(dataloaders, args.property)
 
# --- Model creation ---
# use_sparse: whether any sparse edge scheduling is needed at all
use_sparse = args.pairwise or args.sparse or args.hybrid
 
# n_sparse_layers: how many layers the sparse schedule must cover.
# For HybridEGNN this is the full total (standard + pairwise) so that
# assemble_batch_sparse_edges returns one entry per layer and forward()
# can index at n_standard_layers + i correctly.
# For PairwiseEGNN / SparseEGNN it is just n_layers.
if args.hybrid:
    n_sparse_layers = args.n_standard_layers + args.n_pairwise_layers
else:
    n_sparse_layers = args.n_layers
 
if args.pairwise:
    pw_nf = args.nf_sparse if args.nf_sparse is not None else args.nf
    model = PairwiseEGNN(
        in_node_nf=15, in_edge_nf=0, hidden_nf=pw_nf, device=device,
        n_layers=args.n_layers, coords_weight=1.0,
        attention=args.attention, node_attr=args.node_attr,
        frame_ordering=args.frame_ordering, frame_scoring=args.frame_scoring)
    model_name = 'PairwiseEGNN'
    hidden_nf_used = pw_nf
    print(f"Model: PairwiseEGNN | ordering={args.frame_ordering} | scoring={args.frame_scoring} | "
          f"hidden_nf={pw_nf} | layers={args.n_layers} | vizing_coloring={args.use_vizing_coloring}")
 
elif args.sparse:
    sp_nf = args.nf_sparse if args.nf_sparse is not None else args.nf
    model = SparseEGNN(
        in_node_nf=15, in_edge_nf=0, hidden_nf=sp_nf, device=device,
        n_layers=args.n_layers, coords_weight=1.0,
        attention=args.attention, node_attr=args.node_attr,
        frame_ordering=args.frame_ordering, frame_scoring=args.frame_scoring)
    model_name = 'SparseEGNN'
    hidden_nf_used = sp_nf
    print(f"Model: SparseEGNN | ordering={args.frame_ordering} | scoring={args.frame_scoring} | "
          f"hidden_nf={sp_nf} | layers={args.n_layers} | vizing_coloring={args.use_vizing_coloring}")
 
elif args.hybrid:
    pairwise_nf = args.nf_sparse if args.nf_sparse is not None else args.nf

    model = HybridEGNN(
        in_node_nf=15,
        in_edge_nf=0,
        hidden_nf=args.nf,
        pairwise_nf=pairwise_nf,
        device=device,
        n_standard_layers=args.n_standard_layers,
        n_pairwise_layers=args.n_pairwise_layers,
        coords_weight=1.0,
        attention=args.attention,
        node_attr=args.node_attr,
        frame_ordering=args.frame_ordering,
        frame_scoring=args.frame_scoring,
        pairwise_layer_type=args.pairwise_layer_type)

    model_name = 'HybridEGNN'

    hidden_nf_used = args.nf
    print(f"Model: HybridEGNN | standard_layers={args.n_standard_layers} | "
      f"pairwise_layer_type={args.pairwise_layer_type} | "
      f"pairwise_layers={args.n_pairwise_layers} | total={n_sparse_layers} | "
      f"ordering={args.frame_ordering} | scoring={args.frame_scoring} | "
      f"hidden_nf={args.nf} | pairwise_nf={pairwise_nf} | "
      f"vizing_coloring={args.use_vizing_coloring}")
 
else:
    model = EGNN(
        in_node_nf=15, in_edge_nf=0, hidden_nf=args.nf, device=device,
        n_layers=args.n_layers, coords_weight=1.0,
        attention=args.attention, node_attr=args.node_attr)
    model_name = 'EGNN'
    hidden_nf_used = args.nf
    print(f"Model: EGNN (standard) | layers={args.n_layers}")
 
# Swap in the fused CUDA dense layers. Same rationale as the pairwise swap:
# done after construction so initialisation and the RNG stream are untouched.
# These layers are shared by EGNN and HyEGNN, so this speeds up both equally.
if args.fused_dense:
    sys.path.insert(0, os.path.join(os.path.dirname(EGNN_ROOT), 'src'))
    from EGNN import FusedEGCLMask
    # mask_is_ones is only safe while collate emits compressed edges (all-ones
    # mask). EGNN_BASELINE disables that A/B switch and restores the real 0/1 mask.
    compressed = not os.environ.get('EGNN_BASELINE')
    n_dense = args.n_standard_layers if args.hybrid else args.n_layers
    for i in range(n_dense):
        eager_layer = model._modules[f"gcl_{i}"]
        model._modules[f"gcl_{i}"] = FusedEGCLMask(
            args.nf, args.nf, args.nf, attention=bool(args.attention),
            mask_is_ones=compressed).to(device).load_from(eager_layer)
    print(f"Using fused CUDA dense layers (src/EGNN) for {n_dense} layers")

# Swap in the fused CUDA pairwise layers. Done AFTER construction so parameter
# initialisation (and therefore the RNG stream) is identical to the eager path;
# the fused modules just take a copy of the weights that were just created.
if args.fused_pairwise:
    if not (args.hybrid and args.pairwise_layer_type == 'sym_asym'):
        parser.error("--fused_pairwise requires --hybrid with --pairwise_layer_type sym_asym")
    sys.path.insert(0, os.path.join(os.path.dirname(EGNN_ROOT), 'src'))
    from HyEGNN import FusedPairwiseSymAsymLayer
    from HyEGNN.fused_pairwise_mlp import FusedPairwiseSymAsymMLP
    # nf=64: fully fused (MLPs in shared memory, ~13 launches/layer).
    # other nf: prologue/epilogue kernels around cuBLAS (~101 launches/layer).
    Fused = FusedPairwiseSymAsymMLP if pairwise_nf == 64 else FusedPairwiseSymAsymLayer
    for i in range(args.n_pairwise_layers):
        eager_layer = model._modules[f"pairwise_{i}"]
        model._modules[f"pairwise_{i}"] = Fused(pairwise_nf).to(device).load_from(eager_layer)
    print(f"Using {Fused.__name__} (src/HyEGNN) for {args.n_pairwise_layers} pairwise layers")

if args.cuda_graphs:
    from qm9.cuda_graphs import make_linears_graph_safe
    make_linears_graph_safe(model)
    print("Using graph-safe GEMM + bias linear layers")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters: {total_params:,} (trainable: {trainable_params:,})")
 
layer_info = {}
for name, module in model.named_modules():
    n = sum(p.numel() for p in module.parameters(recurse=False))
    if n > 0:
        layer_info[name] = n
print(f"Per-layer params: {json.dumps(layer_info, indent=2)}")
 
# --- Precompute colorings ---
coloring_cache = None
precompute_time = 0.0
if use_sparse:
    print('Precomputing molecule colorings (one-time cost)...')
    t0 = time.time()
    coloring_cache = precompute_molecule_colorings(
        dataloaders=dataloaders,
        n_layers=n_sparse_layers,          # FIX: full layer count for HybridEGNN
        frame_ordering=args.frame_ordering,
        frame_scoring=args.frame_scoring,
        use_vizing_coloring=args.use_vizing_coloring,
    )
    precompute_time = time.time() - t0
    print(f"Precompute time: {precompute_time:.2f}s")

    # Move the per-batch sparse edge assembly into the DataLoader workers. It is
    # ~10 ms of pure Python per batch and was sitting on the critical path
    # between fetching a batch and launching the forward pass; in a worker it
    # overlaps with GPU compute. The rows/cols produced are identical, so this
    # changes speed only. Needs the coloring cache, hence after the precompute.
    if args.num_workers > 0:
        dataloaders = dataset.rebuild_dataloaders(
            dataloaders, args.batch_size, args.num_workers,
            SparseEdgeCollate(_base_collate, coloring_cache, n_sparse_layers,
                              skip_first=args.n_standard_layers if args.hybrid else 0))
        print(f"Sparse edge assembly moved into {args.num_workers} dataloader workers")

optimizer_lr = (torch.tensor(args.lr, device=device)
                if args.cuda_graphs else args.lr)
optimizer = optim.Adam(model.parameters(), lr=optimizer_lr,
                       weight_decay=args.weight_decay, fused=args.fused_adam,
                       capturable=args.cuda_graphs)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
loss_l1 = nn.L1Loss()

cuda_graph_runner = None
if args.cuda_graphs:
    from qm9.cuda_graphs import BucketedCUDAGraphRunner
    cuda_graph_runner = BucketedCUDAGraphRunner(
        model, optimizer, loss_l1, meann, mad, args.batch_size,
        n_sparse_layers, sparse_start=args.n_standard_layers, amp=args.amp)
    print("Using bucketed CUDA graphs for complete HyEGNN training steps")


class _Phase:
    """Per-phase host wall time + GPU time via CUDA events, no syncs until report."""
    def __init__(self, names):
        self.names = names; self.cpu = {n: 0.0 for n in names}; self.ev = {n: [] for n in names}; self.n = 0
    def start(self, name):
        e = torch.cuda.Event(enable_timing=True); e.record(); self._t = time.perf_counter(); self._e = e; self._name = name
    def stop(self):
        e = torch.cuda.Event(enable_timing=True); e.record()
        self.cpu[self._name] += time.perf_counter() - self._t; self.ev[self._name].append((self._e, e))
    def report(self, wall_total):
        torch.cuda.synchronize()
        print(f"\n=== phase profile over {self.n} batches (ms/batch) ===")
        print(f"{'phase':<22}{'CPU issue':>11}{'GPU exec':>10}   note")
        for nm in self.names:
            cpu = self.cpu[nm] / self.n * 1000
            gpu = sum(a.elapsed_time(b) for a, b in self.ev[nm]) / self.n
            note = 'HOST-BOUND' if cpu > gpu * 1.15 and cpu > 0.3 else ''
            print(f"{nm:<22}{cpu:>11.3f}{gpu:>10.3f}   {note}")
        print(f"{'wall per batch':<22}{wall_total / self.n * 1000:>11.3f}")

_PROF_N = int(os.environ.get('EGNN_PROFILE', '0'))


def train(epoch, loader, partition='train'):
    res = {'loss': 0, 'counter': 0, 'loss_arr': [],
           'mae_sum': 0.0, 'mse_sum': 0.0, 'rel_mse_sum': 0.0}
    acc = {k: torch.zeros((), device=device) for k in ('loss', 'mae', 'mse', 'rel_mse')}

    prof = _Phase(['loader_wait', 'h2d_prep', 'edges', 'forward', 'loss', 'backward',
                   'optimizer', 'metrics']) if (_PROF_N and partition == 'train' and epoch == 0) else None
    _t_prev_end = time.perf_counter(); _t_wall0 = _t_prev_end
    for i, data in enumerate(loader):
        graph_training = partition == 'train' and cuda_graph_runner is not None
        if prof:
            prof.cpu['loader_wait'] += time.perf_counter() - _t_prev_end
        if partition == 'train':
            model.train()
            if not graph_training:
                optimizer.zero_grad()
        else:
            model.eval()
        if prof: prof.start('h2d_prep')
 
        batch_size, n_nodes, _ = data['positions'].size()
        # non_blocking pairs with the loader's pin_memory: without it the pinned
        # staging buffers buy nothing. Consumers are on the same stream, so the
        # copies are still ordered before first use.
        atom_positions = data['positions'].view(batch_size * n_nodes, -1).to(device, dtype, non_blocking=True)
        atom_mask = data['atom_mask'].view(batch_size * n_nodes, -1).to(device, dtype, non_blocking=True)
        one_hot = data['one_hot'].to(device, dtype, non_blocking=True)
        charges = data['charges'].to(device, dtype, non_blocking=True)
        nodes = qm9_utils.preprocess_input(one_hot, charges, args.charge_power, charge_scale, device)
        nodes = nodes.view(batch_size * n_nodes, -1)
        if 'dense_rows' in data:
            # Worker-compressed edge index: padded/self edges already removed, so
            # the mask is all ones and never has to cross the PCIe bus.
            edges = [data['dense_rows'].to(device, non_blocking=True),
                     data['dense_cols'].to(device, non_blocking=True)]
            edge_mask = (None if args.fused_dense else
                         torch.ones(edges[0].size(0), 1,
                                    device=device, dtype=dtype))
        else:
            edges = qm9_utils.get_adj_matrix(n_nodes, batch_size, device)
            edge_mask = data['edge_mask'].to(
                device, dtype, non_blocking=True)
        label = data[args.property].to(device, dtype, non_blocking=True)
        if prof: prof.stop(); prof.start('edges')
 
        sparse_edges = None
        if use_sparse and coloring_cache is not None:
            if 'sparse_rows' in data:
                # Already assembled in a dataloader worker; just move to GPU.
                # The third element is discarded by every pairwise layer
                # (`sparse_rows, sparse_cols, _ = ...`), so don't allocate it.
                sparse_edges = [
                    (r.to(device, non_blocking=True),
                     c.to(device, non_blocking=True),
                     None)
                    for r, c in zip(data['sparse_rows'], data['sparse_cols'])
                ]
            else:
                sparse_edges = assemble_batch_sparse_edges(
                    coloring_cache=coloring_cache,
                    charges_batch=data['charges'],
                    atom_mask_batch=data['atom_mask'],
                    n_nodes=n_nodes,
                    n_layers=n_sparse_layers,  # FIX: full layer count for HybridEGNN
                    device=device,
                )
 
        if prof: prof.stop(); prof.start('forward')
        if graph_training:
            # Input staging and metric reductions remain eager.  The replay
            # performs forward, normalized loss, backward, zero_grad, and Adam.
            loss, pred_real = cuda_graph_runner.run(
                nodes, atom_positions, atom_mask, edges, edge_mask, label,
                n_nodes,
                sparse_edges)
            if prof: prof.stop()
        else:
            with torch.set_grad_enabled(partition == 'train'), \
                 torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.amp):
                pred = model(h0=nodes, x=atom_positions, edges=edges,
                             edge_attr=None, node_mask=atom_mask,
                             edge_mask=edge_mask, n_nodes=n_nodes,
                             sparse_edges_per_layer=sparse_edges)
                pred = pred.float()  # keep loss/metrics in fp32 under autocast
                if prof: prof.stop()

                if partition == 'train':
                    if prof: prof.start('loss')
                    loss = loss_l1(pred, (label - meann) / mad)
                    if prof: prof.stop(); prof.start('backward')
                    loss.backward()
                    if prof: prof.stop(); prof.start('optimizer')
                    optimizer.step()
                    if prof: prof.stop()
                    pred_real = mad * pred + meann
                else:
                    pred_real = mad * pred + meann
                    loss = loss_l1(pred_real, label)

        # Metrics accumulate ON THE GPU. The original did five .item() calls
        # here -- five device syncs per batch that drained the launch queue and
        # stopped the CPU running ahead of the GPU. Same formulas, same printed
        # numbers; the host reads them once per log interval and once per epoch.
        if prof: prof.start('metrics')
        res['counter'] += batch_size
        with torch.no_grad():
            loss_d = loss.detach()
            diff = pred_real - label
            acc['loss'] += loss_d * batch_size
            acc['mae'] += diff.abs().sum()
            mse_b = (diff * diff).sum()
            acc['mse'] += mse_b
            acc['rel_mse'] += (mse_b / ((label * label).sum() + 1e-10)) * batch_size
            # A graph bucket reuses one output address.  Preserve the historical
            # values used by logging before a later replay overwrites it.
            res['loss_arr'].append(loss_d.clone() if graph_training else loss_d)
 
        prefix = '' if partition == 'train' else f'>> {partition}\t'
        if i % args.log_interval == 0:
            recent = torch.stack(res['loss_arr'][-10:]).mean().item()   # one sync
            print(prefix + 'Epoch %d \t Iteration %d \t loss %.4f' % (epoch, i, recent))
        if prof:
            prof.stop(); prof.n += 1; _t_prev_end = time.perf_counter()
            if prof.n == _PROF_N:
                prof.report(time.perf_counter() - _t_wall0); prof = None
 
    if partition == 'train':
        lr_scheduler.step()
 
    n = res['counter']
    # single sync for the epoch's totals
    res['loss'], res['mae_sum'], res['mse_sum'], res['rel_mse_sum'] = (
        v.item() for v in (acc['loss'], acc['mae'], acc['mse'], acc['rel_mse']))
    res['loss_arr'] = [float(v) for v in torch.stack(res['loss_arr']).tolist()] if res['loss_arr'] else []
    return res['loss'] / n, res['mae_sum'] / n, res['mse_sum'] / n, res['rel_mse_sum'] / n
 
if __name__ == '__main__':
    arch_info = {
        'model_name': model_name,
        'n_layers': args.n_layers,
        'hidden_nf': hidden_nf_used,
        'in_node_nf': 15,
        'in_edge_nf': 0,
        'attention': args.attention,
        'node_attr': args.node_attr,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'layer_params': layer_info,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'weight_decay': args.weight_decay,
        'property': args.property,
        'charge_power': args.charge_power,
        'dataset_paper': args.dataset_paper,
        'pairwise': args.pairwise,
        'sparse': args.sparse,
        'hybrid': args.hybrid,
        'frame_ordering': args.frame_ordering,
        'frame_scoring': args.frame_scoring,
        'use_vizing_coloring': args.use_vizing_coloring,
        'train_fraction': args.train_fraction,
        'amp': args.amp,
        'tf32': args.tf32,
        'fused_dense': args.fused_dense,
        'fused_pairwise': args.fused_pairwise,
        'fused_adam': args.fused_adam,
        'cuda_graphs': args.cuda_graphs,
    }

    if args.hybrid:
        arch_info['n_standard_layers'] = args.n_standard_layers
        arch_info['n_pairwise_layers'] = args.n_pairwise_layers
        arch_info['n_sparse_layers'] = n_sparse_layers

    if use_sparse:
        arch_info['precompute_time'] = precompute_time

    if cuda_graph_runner is not None:
        arch_info['cuda_graph_dense_quantum'] = cuda_graph_runner.dense_quantum
        arch_info['cuda_graph_max_buckets'] = cuda_graph_runner.max_buckets

    with open(
        args.outf + '/' + args.exp_name + '/architecture.json',
        'w'
    ) as f:
        json.dump(arch_info, f, indent=4)


    # ============================================================
    # RESULTS STORAGE
    # ============================================================

    res = {
        'epochs': [],
        'losess': [],

        # Validation/test performance
        'test_mae': [],
        'test_mse': [],
        'test_rel_mse': [],

        'val_mae': [],
        'val_mse': [],
        'val_rel_mse': [],

        # Best validation/test performance
        'best_val': 1e10,
        'best_test': 1e10,
        'best_epoch': 0,
        'best_test_mae': 1e10,
        'best_test_mse': 1e10,
        'best_test_rel_mse': 1e10,

        # Training MSE
        'train_mse': [],
        'best_train_mse': 1e10,
        'best_train_mse_epoch': 0,
        'best_train_mse_time': 0.0,

        # Timing
        'train_time_total': 0.0,
        'train_time_per_epoch': [],
        'validation_time_total': 0.0,
        'test_time_total': 0.0,
        'total_time': 0.0,
    }

    if use_sparse:
        res['precompute_time'] = precompute_time
    if cuda_graph_runner is not None:
        res['cuda_graph_captures'] = 0
        res['cuda_graph_eager_fallbacks'] = 0


    # ============================================================
    # SETTINGS
    # ============================================================

    # A new training MSE must improve by at least this amount
    # to be considered a new best.
    mse_tolerance = 0.00005


    # ============================================================
    # START TOTAL TIMER
    # ============================================================

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    total_start_time = time.time()


    # ============================================================
    # TRAINING LOOP
    # ============================================================

    for epoch in range(args.epochs):

        # --------------------------------------------------------
        # Training
        # --------------------------------------------------------

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.time()

        train_loss, train_mae, train_mse, train_rel_mse = train(
            epoch,
            dataloaders['train'],
            'train'
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        epoch_train_time = time.time() - t0


        # --------------------------------------------------------
        # Record training timing
        # --------------------------------------------------------

        res['train_time_total'] += epoch_train_time

        res['train_time_per_epoch'].append(
            epoch_train_time
        )


        # --------------------------------------------------------
        # Store training MSE
        # --------------------------------------------------------

        res['train_mse'].append(train_mse)


        # --------------------------------------------------------
        # Check for best training MSE
        # --------------------------------------------------------

        if train_mse < res['best_train_mse'] - mse_tolerance:

            res['best_train_mse'] = train_mse

            res['best_train_mse_epoch'] = epoch

            res['best_train_mse_time'] = (
                res['train_time_total']
            )


        # ========================================================
        # VALIDATION / TEST
        # ========================================================

        if epoch % args.test_interval == 0:

            # ----------------------------------------------------
            # Validation
            # ----------------------------------------------------

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t0 = time.time()

            val_loss, val_mae, val_mse, val_rel_mse = train(
                epoch,
                dataloaders['valid'],
                'valid'
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            validation_time = time.time() - t0

            res['validation_time_total'] += validation_time


            # ----------------------------------------------------
            # Test
            # ----------------------------------------------------

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t0 = time.time()

            test_loss, test_mae, test_mse, test_rel_mse = train(
                epoch,
                dataloaders['test'],
                'test'
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            test_time = time.time() - t0

            res['test_time_total'] += test_time


            # ----------------------------------------------------
            # Store validation/test results
            # ----------------------------------------------------

            res['epochs'].append(epoch)

            res['losess'].append(test_loss)

            res['test_mae'].append(test_mae)
            res['test_mse'].append(test_mse)
            res['test_rel_mse'].append(test_rel_mse)

            res['val_mae'].append(val_mae)
            res['val_mse'].append(val_mse)
            res['val_rel_mse'].append(val_rel_mse)


            # ----------------------------------------------------
            # Best validation/test performance
            # ----------------------------------------------------

            if val_loss < res['best_val']:

                res['best_val'] = val_loss
                res['best_test'] = test_loss
                res['best_epoch'] = epoch

                res['best_test_mae'] = test_mae
                res['best_test_mse'] = test_mse
                res['best_test_rel_mse'] = test_rel_mse


            # ----------------------------------------------------
            # Print results
            # ----------------------------------------------------

            print(
                'Val loss: %.4f \t test loss: %.4f \t epoch %d'
                % (val_loss, test_loss, epoch)
            )

            print(
                '  Train MSE: %.6f \t Test MAE: %.6f \t '
                'MSE: %.6f \t RelMSE: %.6f'
                % (
                    train_mse,
                    test_mae,
                    test_mse,
                    test_rel_mse
                )
            )

            print(
                'Best: val loss: %.4f \t test loss: %.4f \t epoch %d'
                % (
                    res['best_val'],
                    res['best_test'],
                    res['best_epoch']
                )
            )

            print(
                'Best train MSE: %.6f \t epoch %d'
                % (
                    res['best_train_mse'],
                    res['best_train_mse_epoch']
                )
            )

            print(
                '  Train time: %.1fs \t '
                'Validation time: %.1fs \t '
                'Test time: %.1fs'
                % (
                    res['train_time_total'],
                    res['validation_time_total'],
                    res['test_time_total']
                )
            )

            if use_sparse:
                print(
                    '  Precompute time (one-time): %.1fs'
                    % precompute_time
                )


        # ========================================================
        # TOTAL TIME + SAVE RESULTS
        # ========================================================

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        res['total_time'] = (
            time.time() - total_start_time
        )

        if cuda_graph_runner is not None:
            res['cuda_graph_captures'] = cuda_graph_runner.captures
            res['cuda_graph_eager_fallbacks'] = (
                cuda_graph_runner.eager_fallbacks)


        with open(
            args.outf + '/' + args.exp_name + '/losess.json',
            'w'
        ) as outfile:
            json.dump(res, outfile, indent=4)

    dataset.shutdown_dataloaders(dataloaders)

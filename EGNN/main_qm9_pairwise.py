from qm9 import dataset
from qm9.models_pairwise import EGNN, PairwiseEGNN, SparseEGNN, HybridEGNN
from qm9.models_pairwise import precompute_molecule_colorings, assemble_batch_sparse_edges
import torch
from torch import nn, optim
import argparse
from qm9 import utils as qm9_utils
import utils
import json
import time

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
parser.add_argument('--outf', type=str, default='qm9/logs')
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
 
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
loss_l1 = nn.L1Loss()
 
 
def train(epoch, loader, partition='train'):
    res = {'loss': 0, 'counter': 0, 'loss_arr': [],
           'mae_sum': 0.0, 'mse_sum': 0.0, 'rel_mse_sum': 0.0}
 
    for i, data in enumerate(loader):
        if partition == 'train':
            model.train()
            optimizer.zero_grad()
        else:
            model.eval()
 
        batch_size, n_nodes, _ = data['positions'].size()
        atom_positions = data['positions'].view(batch_size * n_nodes, -1).to(device, dtype)
        atom_mask = data['atom_mask'].view(batch_size * n_nodes, -1).to(device, dtype)
        edge_mask = data['edge_mask'].to(device, dtype)
        one_hot = data['one_hot'].to(device, dtype)
        charges = data['charges'].to(device, dtype)
        nodes = qm9_utils.preprocess_input(one_hot, charges, args.charge_power, charge_scale, device)
        nodes = nodes.view(batch_size * n_nodes, -1)
        edges = qm9_utils.get_adj_matrix(n_nodes, batch_size, device)
        label = data[args.property].to(device, dtype)
 
        sparse_edges = None
        if use_sparse and coloring_cache is not None:
            sparse_edges = assemble_batch_sparse_edges(
                coloring_cache=coloring_cache,
                charges_batch=data['charges'],
                atom_mask_batch=data['atom_mask'],
                n_nodes=n_nodes,
                n_layers=n_sparse_layers,  # FIX: full layer count for HybridEGNN
                device=device,
            )
 
        with torch.set_grad_enabled(partition == 'train'):
            pred = model(h0=nodes, x=atom_positions, edges=edges, edge_attr=None,
                         node_mask=atom_mask, edge_mask=edge_mask, n_nodes=n_nodes,
                         sparse_edges_per_layer=sparse_edges)
 
            if partition == 'train':
                loss = loss_l1(pred, (label - meann) / mad)
                loss.backward()
                optimizer.step()
                pred_real = mad * pred + meann
            else:
                pred_real = mad * pred + meann
                loss = loss_l1(pred_real, label)
 
        res['loss'] += loss.item() * batch_size
        res['counter'] += batch_size
        res['loss_arr'].append(loss.item())
 
        with torch.no_grad():
            mae_val = torch.abs(pred_real - label).sum().item()
            mse_val = ((pred_real - label) ** 2).sum().item()
            denom = (label ** 2).sum().item() + 1e-10
            rel_mse_val = mse_val / denom
            res['mae_sum'] += mae_val
            res['mse_sum'] += mse_val
            res['rel_mse_sum'] += rel_mse_val * batch_size
 
        prefix = '' if partition == 'train' else f'>> {partition}\t'
        if i % args.log_interval == 0:
            print(prefix + 'Epoch %d \t Iteration %d \t loss %.4f' % (
                epoch, i, sum(res['loss_arr'][-10:]) / len(res['loss_arr'][-10:])))
 
    if partition == 'train':
        lr_scheduler.step()
 
    n = res['counter']
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
    }

    if args.hybrid:
        arch_info['n_standard_layers'] = args.n_standard_layers
        arch_info['n_pairwise_layers'] = args.n_pairwise_layers
        arch_info['n_sparse_layers'] = n_sparse_layers

    if use_sparse:
        arch_info['precompute_time'] = precompute_time

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


        with open(
            args.outf + '/' + args.exp_name + '/losess.json',
            'w'
        ) as outfile:
            json.dump(res, outfile, indent=4)
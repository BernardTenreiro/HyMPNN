python -u EGNN/main_qm9_pairwise.py --num_workers 0 --lr 5e-4 --property homo --exp_name has_128_64 --epochs 1000 --batch_size 128 --nf 128 --nf_sparse 64 --n_layers 10 --n_standard_layers 5 --n_pairwise_layers 5 --hybrid --pairwise_layer_type sym_asym
python -u EGNN/main_qm9_pairwise.py --num_workers 0 --lr 5e-4 --property homo --exp_name has_64_64 --epochs 1000 --batch_size 128 --nf 64 --nf_sparse 64 --n_layers 10 --n_standard_layers 5 --n_pairwise_layers 5 --hybrid --pairwise_layer_type sym_asym

python -u EGNN/main_qm9_pairwise.py --num_workers 0 --lr 5e-4 --property homo --exp_name he_128_64 --epochs 1000 --batch_size 128 --nf 128 --nf_sparse 64 --n_layers 10 --n_standard_layers 5 --n_pairwise_layers 5 --hybrid --pairwise_layer_type egcl
python -u EGNN/main_qm9_pairwise.py --num_workers 0 --lr 5e-4 --property homo --exp_name he_64_64 --epochs 1000 --batch_size 128 --nf 64 --nf_sparse 64 --n_layers 10 --n_standard_layers 5 --n_pairwise_layers 5 --hybrid --pairwise_layer_type egcl

python -u EGNN/main_qm9_pairwise.py --num_workers 0 --lr 5e-4 --property homo --exp_name hs_128_64 --epochs 1000 --batch_size 128 --nf 128 --nf_sparse 64 --n_layers 10 --n_standard_layers 5 --n_pairwise_layers 5 --hybrid --pairwise_layer_type symmetric
python -u EGNN/main_qm9_pairwise.py --num_workers 0 --lr 5e-4 --property homo --exp_name hs_64_64 --epochs 1000 --batch_size 128 --nf 64 --nf_sparse 64 --n_layers 10 --n_standard_layers 5 --n_pairwise_layers 5 --hybrid --pairwise_layer_type symmetric

python -u EGNN/main_qm9_pairwise.py --num_workers 0 --lr 5e-4 --property homo --exp_name hj_128_64 --epochs 1000 --batch_size 128 --nf 128 --nf_sparse 64 --n_layers 10 --n_standard_layers 5 --n_pairwise_layers 5 --hybrid --pairwise_layer_type joint
python -u EGNN/main_qm9_pairwise.py --num_workers 0 --lr 5e-4 --property homo --exp_name hj_64_64 --epochs 1000 --batch_size 128 --nf 64 --nf_sparse 64 --n_layers 10 --n_standard_layers 5 --n_pairwise_layers 5 --hybrid --pairwise_layer_type joint

python -u EGNN/main_qm9_pairwise.py --num_workers 0 --lr 5e-4 --property homo --exp_name s5_128 --epochs 1000 --batch_size 128 --nf 128 --n_layers 5
python -u EGNN/main_qm9_pairwise.py --num_workers 0 --lr 5e-4 --property homo --exp_name s5_64--epochs 1000 --batch_size 128 --nf 64 --n_layers 5
python -u EGNN/main_qm9_pairwise.py --num_workers 0 --lr 5e-4 --property homo --exp_name s7_128 --epochs 1000 --batch_size 128 --nf 128 --n_layers 7
python -u EGNN/main_qm9_pairwise.py --num_workers 0 --lr 5e-4 --property homo --exp_name s7_64 --epochs 1000 --batch_size 128 --nf 64 --n_layers 7
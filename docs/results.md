# Results – Standard vs Sparse EGNN on QM9 (alpha)

## Standard EGNN (7L, 1000 epochs)
- Final val loss: 0.0670
- Final test loss: 0.0682
- Final test MAE: 0.068192
- Final RelMSE: 0.000008
- Best val loss: 0.0669
- Best test loss: 0.0683
- Best epoch: 865
- Train time: 53412.1s
- Test time: 13464.6s

## Sparse EGNN (7L, 1000 epochs)
- Final val loss: 1.0975
- Final test loss: 1.0965
- Final test MAE: 1.096488
- Final RelMSE: 0.000518
- Best val loss: 1.0968
- Best test loss: 1.0970
- Best epoch: 795
- Train time: 61529.5s
- Test time: 18125.5s
- Precompute time: 72.5s

## Sparse EGNN (14L, partial run to epoch 820)
- Final val loss: 0.8399
- Final test loss: 0.8563
- Final test MAE: 0.856297
- Final RelMSE: 0.000350
- Best val loss: 0.8372
- Best test loss: 0.8551
- Best epoch: 669
- Train time: 72467.5s
- Test time: 20396.4s
- Precompute time: 75.5s
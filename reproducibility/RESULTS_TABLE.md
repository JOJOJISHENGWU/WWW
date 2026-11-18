# Results Traceability Table (Anonymous Review Package)

This table records the released verification artifact in this anonymous repository.
Only PEMS08 is provided here to avoid exposing non-public engineering assets.

| Dataset | Config File | Checkpoint File | Log File | Best Val MAE | Test MAE | Test RMSE | Test MAPE | Notes |
|---------|-------------|-----------------|----------|--------------|----------|-----------|-----------|-------|
| PEMS08 | `configs/pems08.yaml` | `checkpoints/pems08_best.pth` | `logs/pems08_train_full.log` | 11.9467 | 11.8250 | 19.3114 | 6.69% | Early stopping at epoch 43; test metrics are reported after loading best-validation checkpoint. |

## Metric Interpretation

- `Best Val MAE`: model-selection metric on validation split.
- `Test MAE/RMSE/MAPE`: final report metrics on held-out test split.

These values are expected to differ because they are computed on different splits.

## Verification Command

```bash
python eval.py --config configs/pems08.yaml --ckpt checkpoints/pems08_best.pth
```

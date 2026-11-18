# Reproducibility Guide (Anonymous Review Package)

This release is a **review-oriented package**: it provides paper-aligned method pseudocode, reproducibility configs, one released checkpoint, and one full training log for independent verification.

## 1) Scope of This Anonymous Release

To protect non-public engineering assets while preserving scientific verifiability:

- Released for direct verification in this package:
  - **PEMS08 checkpoint**: `checkpoints/pems08_best.pth`
  - **PEMS08 full training log**: `logs/pems08_train_full.log`
  - **PEMS08 config**: `configs/pems08.yaml`
  - **Evaluation entrypoint**: `eval.py`
- Not fully released in this package:
  - complete production training stack for all datasets

This policy is aligned with the paper's anonymous review constraints.

## 2) Environment

- OS: Linux / Windows
- Python: >= 3.9
- PyTorch: >= 2.0
- CUDA: >= 11.8 (optional, CPU fallback supported)

Install dependencies:

```bash
pip install -r requirements.txt
```

## 3) One-Command Verification (PEMS08)

```bash
python eval.py --config configs/pems08.yaml --ckpt checkpoints/pems08_best.pth
```

Expected output fields:

- MAE
- RMSE
- MAPE

Reference values are recorded in `reproducibility/RESULTS_TABLE.md`.

## 4) Experimental Setting Alignment with Paper

For the released PEMS08 result, the setting follows the paper's protocol family:

- Chronological split ratio: 7:2:1
- Metrics: MAE / RMSE / MAPE
- Optimization family: Adam-style optimizer and early stopping strategy
- Sequence modeling setting follows the SCOUT pseudocode pipeline and reviewer package config

## 5) Split-Wise Metric Interpretation

The log contains both validation-best and test metrics:

- `Best Val MAE`: best score on validation split during training
- `Test MAE/RMSE/MAPE`: final score on test split after loading best checkpoint

Differences between validation and test numbers are expected because they are measured on different splits.

## 6) Determinism Notes

- Fix seed to `42` (config default)
- Keep software stack fixed across reruns
- Small fluctuations can occur due to hardware/runtime nondeterminism

## 7) Artifact Traceability

Please use `reproducibility/RESULTS_TABLE.md` as the source-of-truth mapping:

- config -> checkpoint -> log -> reported metrics

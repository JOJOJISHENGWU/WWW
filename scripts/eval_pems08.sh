#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

python eval.py \
  --config configs/pems08.yaml \
  --ckpt checkpoints/pems08_best.pth

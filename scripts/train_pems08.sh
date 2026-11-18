#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs checkpoints

# Demo training entrypoint (replace with full pipeline if available)
python train.py 2>&1 | tee "logs/pems08_train_full.log"

echo "Training finished."
echo "Expected checkpoint: checkpoints/scout_best.pth"
echo "Log: logs/pems08_train_full.log"

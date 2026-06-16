#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

python3 scripts/train_debugging_experiments.py \
  --data data/week04_tlc_trip_duration_smoke_10k.parquet \
  --output-dir outputs \
  --epochs 12 \
  --batch-size 512 \
  --seed 6320 \
  --device auto

python3 scripts/plot_learning_curves.py \
  --history-dir outputs \
  --output-dir outputs/plots

echo "Smoke test complete. See outputs/ and outputs/plots/"

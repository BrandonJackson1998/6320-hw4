# CS 6320 — Assignment 4

**Name:** Brandon Jackson  
**Repo:** NYC TLC training-debugging experiments (Part A) and locked BGG portfolio charter (Part B).

## Contents

| Path | Purpose |
| --- | --- |
| `data/week04_tlc_trip_duration_smoke_10k.parquet` | Local 10k smoke-test dataset |
| `scripts/train_debugging_experiments.py` | Four controlled failure/correction runs |
| `scripts/tlc_common.py` | Parquet loading, splits, scaling helpers |
| `scripts/plot_learning_curves.py` | Required learning-curve PNG generator |
| `run_smoke_test.sh` | Local smoke test (train + plots) |
| `run_week04_smoke_test.slurm` | CHPC smoke-test job |
| `run_week04_tlc_debugging.slurm` | CHPC full-data job |
| `outputs/` | `*_history.csv`, summaries, plots |
| `writeup/CS6320_Assignment4_Jackson.md` | Part A + Part B writeup |

## Setup

```bash
python3 -m pip install -r requirements.txt
```

Or with `uv` (matches CHPC):

```bash
uv sync
```

## Run locally

From repo root:

```bash
bash run_smoke_test.sh
```

Or:

```bash
python3 scripts/train_debugging_experiments.py --data data/week04_tlc_trip_duration_smoke_10k.parquet --output-dir outputs
python3 scripts/plot_learning_curves.py --history-dir outputs --output-dir outputs/plots
```

## Run on CHPC

Submit from repo root on **Granite** (`larsenc` / `utucset-gpu-grn`):

```bash
ssh u1303977@granite.chpc.utah.edu
cd 6320-hw4
mkdir -p logs
sbatch run_week04_smoke_test.slurm
```

Full prepared dataset (stronger evidence):

```bash
sbatch run_week04_tlc_debugging.slurm
```

Staged data default:

```text
/scratch/general/vast/u0090307/ut-cs6320/datasets/week04_tlc_trip_duration_smoke_10k.parquet
/scratch/general/vast/u0090307/ut-cs6320/datasets/week04_tlc_trip_duration_full.parquet
```

Override with `DATA_DIR`, `SMOKE_DATA`, or `FULL_DATA` if needed.

## Experiments (Part A)

| Case | File prefix | Purpose |
| --- | --- | --- |
| 1 failure | `case1_high_lr_failure` | Learning rate `0.2` → unstable validation |
| 1 correction | `case1_lr_corrected` | Learning rate `0.001` |
| 2 failure | `case2_unscaled_target_failure` | No target standardization |
| 2 correction | `case2_scaled_target_corrected` | Standardized log-duration target |

Seed: `6320`. Mini-batch size: `512`. Uses prepared `split` column and categorical embeddings.

## Notes

- Part A uses the Week 4 NYC TLC Parquet task, adapted from the Assignment 3 visible training-loop pattern.
- Part B locks the Board Game Geek portfolio charter from Assignment 3.
- Local smoke file checks code paths; CHPC full file is intended for stronger evidence runs.

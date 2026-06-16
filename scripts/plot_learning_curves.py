#!/usr/bin/env python3
"""Create required learning-curve plots from Week 4 scenario history CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO_DIR = Path(__file__).resolve().parent.parent
DEFAULT_HISTORY_DIR = REPO_DIR / "outputs"
DEFAULT_OUTPUT_DIR = REPO_DIR / "outputs" / "plots"

CASE_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("case1_high_lr_failure", "case1_lr_corrected", "Case 1 — learning rate"),
    ("case2_unscaled_target_failure", "case2_scaled_target_corrected", "Case 2 — target scaling"),
)

SHORT_LABELS = {
    "case1_high_lr_failure": "failure (lr=0.2)",
    "case1_lr_corrected": "corrected (lr=0.001)",
    "case2_unscaled_target_failure": "failure (unscaled target)",
    "case2_scaled_target_corrected": "corrected (scaled target)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Week 4 TLC debugging learning curves.")
    parser.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR, help="Directory containing *_history.csv files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for PNG plots.")
    return parser.parse_args()


def load_history(history_dir: Path, stem: str) -> pd.DataFrame:
    path = history_dir / f"{stem}_history.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing history file: {path}")
    return pd.read_csv(path)


def plot_pair_panel(
    ax: plt.Axes,
    history_dir: Path,
    failure_stem: str,
    corrected_stem: str,
    metric: str,
    ylabel: str,
    *,
    log_y: bool,
) -> None:
    for stem, linestyle in ((failure_stem, "--"), (corrected_stem, "-")):
        frame = load_history(history_dir, stem)
        ax.plot(
            frame["epoch"],
            frame[metric],
            marker="o",
            linewidth=2,
            linestyle=linestyle,
            label=SHORT_LABELS[stem],
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize="small")


def plot_metric_by_pair(
    history_dir: Path,
    metric: str,
    ylabel: str,
    output_path: Path,
    *,
    log_y: bool,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    for ax, (failure_stem, corrected_stem, title) in zip(axes, CASE_PAIRS):
        plot_pair_panel(ax, history_dir, failure_stem, corrected_stem, metric, ylabel, log_y=log_y)
        ax.set_title(title)
    fig.suptitle(ylabel, y=1.01, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    history_files = sorted(args.history_dir.glob("*_history.csv"))
    if not history_files:
        raise FileNotFoundError(f"no *_history.csv files found in {args.history_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_metric_by_pair(
        args.history_dir,
        "train_loss_scaled_mse",
        "Training Loss (scaled MSE)",
        args.output_dir / "training_loss_curves.png",
        log_y=True,
    )
    plot_metric_by_pair(
        args.history_dir,
        "validation_loss_scaled_mse",
        "Validation Loss (scaled MSE)",
        args.output_dir / "validation_loss_curves.png",
        log_y=True,
    )
    plot_metric_by_pair(
        args.history_dir,
        "validation_mae_seconds",
        "Validation MAE (seconds)",
        args.output_dir / "validation_mae_curves.png",
        log_y=True,
    )
    print(f"wrote learning-curve plots to {args.output_dir}")


if __name__ == "__main__":
    main()

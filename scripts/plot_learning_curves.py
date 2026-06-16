#!/usr/bin/env python3
"""Create required learning-curve plots from Week 4 scenario history CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_HISTORY_DIR = PACKAGE_DIR / "outputs"
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "outputs" / "plots"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Week 4 TLC debugging learning curves.")
    parser.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR, help="Directory containing *_history.csv files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for PNG plots.")
    return parser.parse_args()


def plot_metric(history_files: list[Path], metric: str, ylabel: str, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    for history_file in history_files:
        frame = pd.read_csv(history_file)
        label = history_file.name.removesuffix("_history.csv")
        plt.plot(frame["epoch"], frame[metric], marker="o", linewidth=2, label=label)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize="small")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    history_files = sorted(args.history_dir.glob("*_history.csv"))
    if not history_files:
        raise FileNotFoundError(f"no *_history.csv files found in {args.history_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_metric(history_files, "train_loss_scaled_mse", "Training Loss (scaled MSE)", args.output_dir / "training_loss_curves.png")
    plot_metric(history_files, "validation_loss_scaled_mse", "Validation Loss (scaled MSE)", args.output_dir / "validation_loss_curves.png")
    plot_metric(history_files, "validation_mae_seconds", "Validation MAE (seconds)", args.output_dir / "validation_mae_curves.png")
    print(f"wrote learning-curve plots to {args.output_dir}")


if __name__ == "__main__":
    main()

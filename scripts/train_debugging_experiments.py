"""Week 4 TLC training-debugging experiments with visible PyTorch loops."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from tlc_common import (
    CATEGORY_VOCAB_SIZES,
    CATEGORICAL_COLUMNS,
    RANDOM_STATE,
    PreparedSplits,
    default_data_path,
    default_output_dir,
    load_prepared_frame,
    log_to_seconds,
    mae_seconds,
    make_loader,
    prepare_splits,
    scale_targets,
    unscale_predictions,
    write_json,
)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    learning_rate: float
    scale_target: bool
    scale_numeric: bool = True
    hidden_dim: int = 64
    embed_dim: int = 16
    epochs: int = 12
    batch_size: int = 512
    weight_init_scale: float = 1.0
    description: str = ""


EXPERIMENTS: list[ExperimentConfig] = [
    ExperimentConfig(
        name="case1_high_lr_failure",
        learning_rate=0.2,
        scale_target=True,
        description="Failure: learning rate too high for stable Adam updates.",
    ),
    ExperimentConfig(
        name="case1_lr_corrected",
        learning_rate=0.001,
        scale_target=True,
        description="Correction: reduce learning rate while holding other settings fixed.",
    ),
    ExperimentConfig(
        name="case2_unscaled_target_failure",
        learning_rate=0.001,
        scale_target=False,
        description="Failure: optimize raw log-duration targets without standardization.",
    ),
    ExperimentConfig(
        name="case2_scaled_target_corrected",
        learning_rate=0.001,
        scale_target=True,
        description="Correction: standardize log-duration targets using train mean/std.",
    ),
]


class TLCRegressor(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        embed_dim: int,
        weight_init_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.pickup_embed = nn.Embedding(CATEGORY_VOCAB_SIZES["pickup_location_id"], embed_dim)
        self.dropoff_embed = nn.Embedding(CATEGORY_VOCAB_SIZES["dropoff_location_id"], embed_dim)
        self.hour_embed = nn.Embedding(CATEGORY_VOCAB_SIZES["pickup_hour"], max(4, embed_dim // 2))
        self.dow_embed = nn.Embedding(CATEGORY_VOCAB_SIZES["pickup_day_of_week"], max(4, embed_dim // 4))
        input_dim = embed_dim * 2 + max(4, embed_dim // 2) + max(4, embed_dim // 4) + 3
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.output = nn.Linear(hidden_dim, 1)
        self._reset_parameters(weight_init_scale)

    def _reset_parameters(self, scale: float) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
                module.weight.data.mul_(scale)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02 * scale)

    def forward(self, categorical: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        pickup = categorical[:, 0]
        dropoff = categorical[:, 1]
        hour = categorical[:, 2]
        dow = categorical[:, 3]
        features = torch.cat(
            [
                self.pickup_embed(pickup),
                self.dropoff_embed(dropoff),
                self.hour_embed(hour),
                self.dow_embed(dow),
                numeric,
            ],
            dim=1,
        )
        hidden = self.activation(self.hidden(features))
        return self.output(hidden).squeeze(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Week 4 TLC debugging experiments.")
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for all cases.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional experiment name subset.",
    )
    return parser.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def run_epoch_train(
    model: TLCRegressor,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    *,
    target_mean: float,
    target_std: float,
    scale_target: bool,
) -> tuple[float, bool]:
    model.train()
    total_loss = 0.0
    total_rows = 0
    saw_nonfinite = False

    for categorical, numeric, target_log, _ in loader:
        categorical = categorical.to(device)
        numeric = numeric.to(device)
        target = scale_targets(target_log.to(device), target_mean, target_std, scale_target)

        optimizer.zero_grad()
        predictions = model(categorical, numeric)
        loss = loss_fn(predictions, target)

        if not torch.isfinite(loss):
            saw_nonfinite = True
            total_loss += float("nan")
            total_rows += len(target)
            continue

        loss.backward()
        optimizer.step()

        batch_loss = float(loss.item())
        total_loss += batch_loss * len(target)
        total_rows += len(target)

    mean_loss = total_loss / max(total_rows, 1)
    return mean_loss, saw_nonfinite


@torch.no_grad()
def evaluate(
    model: TLCRegressor,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    *,
    target_mean: float,
    target_std: float,
    scale_target: bool,
) -> tuple[float, float, bool]:
    model.eval()
    total_loss = 0.0
    total_rows = 0
    mae_total = 0.0
    saw_nonfinite = False

    for categorical, numeric, target_log, target_seconds in loader:
        categorical = categorical.to(device)
        numeric = numeric.to(device)
        target = scale_targets(target_log.to(device), target_mean, target_std, scale_target)
        seconds = target_seconds.to(device)

        predictions = model(categorical, numeric)
        loss = loss_fn(predictions, target)

        if not torch.isfinite(loss):
            saw_nonfinite = True
            total_loss += float("nan")
            total_rows += len(target)
            mae_total += float("nan") * len(target)
            continue

        pred_log = unscale_predictions(predictions, target_mean, target_std, scale_target)
        pred_seconds = log_to_seconds(pred_log)

        batch_loss = float(loss.item())
        batch_mae = mae_seconds(seconds, pred_seconds)
        total_loss += batch_loss * len(target)
        mae_total += batch_mae * len(target)
        total_rows += len(target)

    mean_loss = total_loss / max(total_rows, 1)
    mean_mae = mae_total / max(total_rows, 1)
    return mean_loss, mean_mae, saw_nonfinite


def save_history(path: Path, history: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "train_loss_scaled_mse",
        "validation_loss_scaled_mse",
        "validation_mae_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({key: row[key] for key in fieldnames})


def run_experiment(
    config: ExperimentConfig,
    prepared: PreparedSplits,
    *,
    device: torch.device,
    output_dir: Path,
    batch_size: int,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = make_loader(prepared.train, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(prepared.validation, batch_size=batch_size, shuffle=False)

    model = TLCRegressor(
        hidden_dim=config.hidden_dim,
        embed_dim=config.embed_dim,
        weight_init_scale=config.weight_init_scale,
    ).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    history: list[dict[str, float]] = []
    prediction_summary: dict[str, float] | None = None

    print(f"\n=== {config.name} ===")
    print(config.description)
    print(f"lr={config.learning_rate}, scale_target={config.scale_target}, scale_numeric={config.scale_numeric}")

    for epoch in range(1, config.epochs + 1):
        train_loss, train_bad = run_epoch_train(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            target_mean=prepared.target_log_mean,
            target_std=prepared.target_log_std,
            scale_target=config.scale_target,
        )
        val_loss, val_mae, val_bad = evaluate(
            model,
            val_loader,
            loss_fn,
            device,
            target_mean=prepared.target_log_mean,
            target_std=prepared.target_log_std,
            scale_target=config.scale_target,
        )

        record = {
            "epoch": epoch,
            "train_loss_scaled_mse": train_loss,
            "validation_loss_scaled_mse": val_loss,
            "validation_mae_seconds": val_mae,
        }
        history.append(record)

        status = ""
        if train_bad or val_bad or (isinstance(train_loss, float) and math.isnan(train_loss)):
            status = " [non-finite detected]"

        print(
            f"epoch={epoch:02d} train_mse={train_loss:.4f} "
            f"val_mse={val_loss:.4f} val_mae={val_mae:.1f}s{status}"
        )

        if train_bad or val_bad:
            print("Stopping early because loss became non-finite.")
            break

    # Prediction collapse check on final epoch
    model.eval()
    with torch.no_grad():
        preds = []
        for categorical, numeric, target_log, _ in val_loader:
            categorical = categorical.to(device)
            numeric = numeric.to(device)
            pred_log = unscale_predictions(
                model(categorical, numeric),
                prepared.target_log_mean,
                prepared.target_log_std,
                config.scale_target,
            )
            preds.append(log_to_seconds(pred_log).cpu().numpy())
        pred_array = np.concatenate(preds)
        prediction_summary = {
            "validation_pred_seconds_mean": float(pred_array.mean()),
            "validation_pred_seconds_std": float(pred_array.std()),
            "validation_pred_seconds_min": float(pred_array.min()),
            "validation_pred_seconds_max": float(pred_array.max()),
        }

    history_path = output_dir / f"{config.name}_history.csv"
    save_history(history_path, history)

    summary = {
        "experiment": config.name,
        "description": config.description,
        "learning_rate": config.learning_rate,
        "scale_target": config.scale_target,
        "scale_numeric": config.scale_numeric,
        "hidden_dim": config.hidden_dim,
        "embed_dim": config.embed_dim,
        "epochs_requested": config.epochs,
        "epochs_ran": len(history),
        "batch_size": batch_size,
        "seed": seed,
        "device": str(device),
        "target_log_mean": prepared.target_log_mean,
        "target_log_std": prepared.target_log_std,
        "final_train_loss_scaled_mse": history[-1]["train_loss_scaled_mse"],
        "final_validation_loss_scaled_mse": history[-1]["validation_loss_scaled_mse"],
        "final_validation_mae_seconds": history[-1]["validation_mae_seconds"],
        "prediction_summary": prediction_summary,
        "history_csv": str(history_path),
    }
    summary_path = output_dir / f"{config.name}_summary.json"
    write_json(summary_path, summary)
    return summary


def main() -> None:
    args = parse_args()
    data_path = args.data or default_data_path()
    output_dir = args.output_dir or default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    frame = load_prepared_frame(data_path)
    configs = EXPERIMENTS
    if args.only:
        wanted = set(args.only)
        configs = [cfg for cfg in EXPERIMENTS if cfg.name in wanted]

    all_summaries = []
    for config in configs:
        prepared = prepare_splits(frame, scale_numeric=config.scale_numeric)
        if args.epochs is not None:
            config = ExperimentConfig(
                name=config.name,
                learning_rate=config.learning_rate,
                scale_target=config.scale_target,
                scale_numeric=config.scale_numeric,
                hidden_dim=config.hidden_dim,
                embed_dim=config.embed_dim,
                epochs=args.epochs,
                batch_size=config.batch_size,
                weight_init_scale=config.weight_init_scale,
                description=config.description,
            )
        summary = run_experiment(
            config,
            prepared,
            device=device,
            output_dir=output_dir,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        all_summaries.append(summary)

    manifest = {
        "data_path": str(data_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "seed": args.seed,
        "device": str(device),
        "experiments": all_summaries,
    }
    write_json(output_dir / "experiment_manifest.json", manifest)
    print(f"\nSaved experiment outputs to {output_dir}")


if __name__ == "__main__":
    main()

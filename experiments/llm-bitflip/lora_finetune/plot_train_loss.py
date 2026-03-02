import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def plot_train_loss(csv_path: Path, output_path: Optional[Path] = None) -> None:
    """Plot train loss vs step and save the figure."""
    df = pd.read_csv(csv_path)

    step_col = "Step"
    loss_col = "Llama-3.1-8B-bitflip-lora-r32 - train_loss"
    baseline_loss = 2.06842

    if step_col not in df.columns:
        raise ValueError(f"Missing column: {step_col}")
    if loss_col not in df.columns:
        raise ValueError(f"Missing column: {loss_col}")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        df[step_col], df[loss_col], label="bitflip r32", color="tab:blue", linewidth=1.8
    )
    ax.axhline(
        baseline_loss, color="tab:red", linestyle="--", linewidth=1.2, label="original"
    )

    ax.set_xlabel("Train step")
    ax.set_ylabel("Train loss")
    ax.set_title("Llama-3.1-8B bitflip lora Fine-tuning")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend()
    fig.tight_layout()

    if output_path is None:
        output_path = csv_path.with_suffix(".png")
        output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=200)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot train loss vs step from a W&B CSV export."
    )
    parser.add_argument("csv", type=Path, help="Path to the W&B CSV export")
    parser.add_argument(
        "--output", "-o", type=Path, default=None, help="Path to save the plot (PNG)"
    )
    args = parser.parse_args()

    plot_train_loss(args.csv, args.output)

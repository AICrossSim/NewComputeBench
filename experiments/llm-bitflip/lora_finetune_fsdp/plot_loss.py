#!/usr/bin/env python3
"""Visualize training loss from a bitflip-LoRA training log.

Parses lines of the form produced by train.py:
    ... [INFO] Step 1000/21000 | Loss: 1.6232 | LR: 9.95e-06 | Tokens/s: 884 | Elapsed: 74142.9s

Usage:
    python plot_loss.py [LOG_FILE] [-o OUT.png] [-w SMOOTH_WINDOW]
"""
import argparse
import re
import sys

import matplotlib

matplotlib.use("Agg")  # headless: write a file, no display needed
import matplotlib.pyplot as plt

LINE_RE = re.compile(
    r"Step\s+(\d+)/(\d+)\s+\|\s+Loss:\s+([\d.]+)\s+\|\s+LR:\s+([\d.eE+-]+)"
)


def parse_log(path):
    """Return (steps, losses, lrs, total_steps) parsed from the log file."""
    steps, losses, lrs = [], [], []
    total_steps = None
    with open(path) as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            steps.append(int(m.group(1)))
            total_steps = int(m.group(2))
            losses.append(float(m.group(3)))
            lrs.append(float(m.group(4)))
    return steps, losses, lrs, total_steps


def moving_average(values, window):
    """Centered simple moving average; returns a list the same length as input."""
    if window <= 1 or len(values) < window:
        return list(values)
    out = []
    half = window // 2
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def main():
    parser = argparse.ArgumentParser(description="Plot training loss from a log file")
    parser.add_argument(
        "log_file", nargs="?", default="logs/train_70b.log",
        help="Path to the training log (default: logs/train_70b.log)",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output image path (default: <log_file>_loss.png)",
    )
    parser.add_argument(
        "-w", "--window", type=int, default=20,
        help="Smoothing window in logged points (default: 20)",
    )
    args = parser.parse_args()

    steps, losses, lrs, total_steps = parse_log(args.log_file)
    if not steps:
        sys.exit(f"No 'Step .. | Loss: ..' lines found in {args.log_file}")

    out = args.output or args.log_file.rsplit(".", 1)[0] + "_loss.png"
    smoothed = moving_average(losses, args.window)

    fig, (ax_loss, ax_lr) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax_loss.plot(steps, losses, color="tab:blue", alpha=0.3, lw=0.8, label="loss (raw)")
    ax_loss.plot(steps, smoothed, color="tab:blue", lw=2,
                 label=f"loss (smoothed, w={args.window})")
    ax_loss.axhline(1.8198, color="tab:green", ls="--", lw=1.2,
                    label="original model (1.8198)")
    ax_loss.axhline(4.4918, color="tab:red", ls="--", lw=1.2,
                    label="bitflip, no LoRA (4.4918)")
    ax_loss.set_ylabel("cross-entropy loss")
    ax_loss.set_title(
        f"Training loss — {args.log_file}  "
        f"(step {steps[-1]}/{total_steps}, latest loss {losses[-1]:.4f})"
    )
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    ax_lr.plot(steps, lrs, color="tab:orange", lw=1.5)
    ax_lr.set_ylabel("learning rate")
    ax_lr.set_xlabel("step")
    ax_lr.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"Parsed {len(steps)} points (step {steps[0]}–{steps[-1]}).")
    print(f"Wrote plot to {out}")


if __name__ == "__main__":
    main()

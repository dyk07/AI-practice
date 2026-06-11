#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot summary of layer-wise reciprocal fit parameters across models.
Reads all results/*/layer_inverse_fit.csv files and plots slope/intercept vs layer for each model on the same figure.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_fit_rows(csv_path: Path):
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "layer": int(row["layer"]),
                    "slope": float(row["slope"]),
                    "intercept": float(row["intercept"]),
                    "r2": float(row.get("r2", "nan")),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot reciprocal-fit slope/intercept vs layer for all models under results/")
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Root results directory")
    parser.add_argument("--output", type=Path, default=Path("results") / "reciprocal_fit_summary.png", help="Output figure path")
    args = parser.parse_args()

    fit_files = sorted(args.results_dir.glob("*/layer_powerlaw_fit.csv"))
    if not fit_files:
        raise SystemExit(f"No layer_inverse_fit.csv files found under {args.results_dir}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=False)
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    for idx, fit_file in enumerate(fit_files):
        model_name = fit_file.parent.name.replace("_", "/")
        rows = load_fit_rows(fit_file)
        layers = [row["layer"] for row in rows]
        slopes = [row["slope"] for row in rows]
        intercepts = [row["intercept"] for row in rows]
        color = colors[idx % len(colors)] if colors else None

        axes[0].plot(layers, slopes, marker="o", linewidth=1.2, label=model_name, color=color)
        axes[1].plot(layers, intercepts, marker="o", linewidth=1.2, label=model_name, color=color)

    axes[0].set_title("Power-law slope vs layer (log-log)")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Slope k in log(y)=k log(x) + log(c)")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(fontsize=8)

    axes[1].set_title("Power-law intercept vs layer (log-log)")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Intercept log(c) in log(y)=k log(x) + log(c)")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(fontsize=8)

    fig.suptitle("Layer-wise power-law (log-log) fit parameters across models")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
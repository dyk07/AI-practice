#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick estimator: convert existing inverse-fit (y = a/x + b) into approximate
power-law fits by sampling synthetic spectra and fitting on log-log axes.
Writes per-model `layer_powerlaw_estimated.csv` and a summary plot.
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def estimate_powerlaw_from_inverse(a, b, n_samples=512, xmin=2):
    xs = np.arange(xmin, xmin + n_samples, dtype=float)
    ys = a / xs + b
    mask = ys > 0
    if mask.sum() < 3:
        return float('nan'), float('nan'), float('nan')
    lx = np.log(xs[mask])
    ly = np.log(ys[mask])
    A = np.column_stack([lx, np.ones_like(lx)])
    coeffs, *_ = np.linalg.lstsq(A, ly, rcond=None)
    k = float(coeffs[0])
    logc = float(coeffs[1])
    fitted = A @ coeffs
    ss_res = float(np.sum((ly - fitted) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 if ss_tot == 0.0 else float(1.0 - ss_res / ss_tot)
    return k, logc, r2


def main():
    results = Path("results")
    inv_files = sorted(results.glob("*/layer_inverse_fit.csv"))
    if not inv_files:
        print("No layer_inverse_fit.csv files found under results/")
        return

    summary = {}
    for f in inv_files:
        model = f.parent.name
        out_csv = f.parent / "layer_powerlaw_estimated.csv"
        rows = []
        with open(f, "r", newline="") as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                layer = int(r["layer"])
                a = float(r["slope"])  # a in a/x + b
                b = float(r["intercept"])  # b
                num = int(r.get("num_points", "512"))
                k, logc, r2 = estimate_powerlaw_from_inverse(a, b, n_samples=min(1024, max(128, num)))
                rows.append((layer, k, logc, r2))
        # save per-model estimated CSV
        with open(out_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["layer", "slope_k", "intercept_logc", "r2"])
            for layer, k, logc, r2 in rows:
                w.writerow([layer, k, logc, r2])
        summary[model] = rows

    # Plot summary: slopes and intercepts
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for idx, (model, rows) in enumerate(sorted(summary.items())):
        layers = [r[0] for r in rows]
        ks = [r[1] for r in rows]
        logcs = [r[2] for r in rows]
        color = colors[idx % len(colors)] if colors else None
        axes[0].plot(layers, ks, marker='o', label=model, color=color)
        axes[1].plot(layers, logcs, marker='o', label=model, color=color)

    axes[0].set_title('Estimated power-law slope k vs layer (from inverse fits)')
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('slope k (log-log)')
    axes[0].grid(True, linestyle='--', alpha=0.4)

    axes[1].set_title('Estimated power-law intercept log(c) vs layer')
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('intercept log(c)')
    axes[1].grid(True, linestyle='--', alpha=0.4)

    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(results / 'powerlaw_estimated_summary.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('Wrote estimated CSVs and results/powerlaw_estimated_summary.png')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Estimate Two-NN intrinsic dimension for 14B model from existing results.

Reads results/*/twonn_intrinsic_dim.csv, interpolates to a common relative-depth grid,
fits intrinsic dimension vs log(model size) at each depth, predicts for 14B,
and writes results/twonn_14B_estimated.csv and results/twonn_14B_estimated.png.
"""
from pathlib import Path
import re
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_size_from_name(name: str) -> float:
    # expects something like Qwen_Qwen3-0.6B or Qwen_Qwen3-14B
    m = re.search(r"-(\d+(?:\.\d+)?)B", name)
    if not m:
        return float('nan')
    return float(m.group(1))


def load_twonn(path: Path):
    data = []
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            layer = int(r['layer'])
            val = float(r['intrinsic_dimension'])
            data.append((layer, val))
    if not data:
        return None
    data.sort()
    layers = np.array([d[0] for d in data], dtype=float)
    vals = np.array([d[1] for d in data], dtype=float)
    return layers, vals


def main():
    results = Path('results')
    files = sorted(results.glob('*/twonn_intrinsic_dim.csv'))
    models = []
    for f in files:
        model = f.parent.name
        size = parse_size_from_name(model)
        loaded = load_twonn(f)
        if loaded is None or np.all(np.isnan(loaded[1])):
            continue
        layers, vals = loaded
        models.append({'name': model, 'size': size, 'layers': layers, 'vals': vals})

    if len(models) < 2:
        print('Not enough models with Two-NN results to estimate 14B')
        return

    # Build relative-depth grid
    grid_n = 100
    rel_grid = np.linspace(0.0, 1.0, grid_n)

    sizes = np.array([m['size'] for m in models], dtype=float)
    log_sizes = np.log(sizes)

    # Interpolate each model onto rel_grid
    interp_vals = []
    for m in models:
        max_layer = m['layers'][-1]
        rel = m['layers'] / max_layer
        # handle NaNs in vals
        vals = m['vals']
        mask = ~np.isnan(vals)
        if mask.sum() < 3:
            interp_vals.append(np.full_like(rel_grid, np.nan, dtype=float))
            continue
        interp = np.interp(rel_grid, rel[mask], vals[mask], left=np.nan, right=np.nan)
        interp_vals.append(interp)

    interp_vals = np.vstack(interp_vals)  # shape (n_models, grid_n)

    # For each grid point, fit linear model intrinsic = alpha + beta * log(size)
    preds = np.full(grid_n, np.nan)
    pred_se = np.full(grid_n, np.nan)
    n_models = interp_vals.shape[0]
    mean_log = log_sizes.mean()
    Sxx = np.sum((log_sizes - mean_log) ** 2)

    for i in range(grid_n):
        y = interp_vals[:, i]
        valid = ~np.isnan(y)
        if valid.sum() < 2:
            continue
        yv = y[valid]
        xs = log_sizes[valid]
        X = np.column_stack([np.ones_like(xs), xs])
        # least squares
        coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
        alpha, beta = coef[0], coef[1]
        # predict for 14B
        log14 = np.log(14.0)
        preds[i] = alpha + beta * log14
        # compute residual std
        resid = yv - (alpha + beta * xs)
        dof = max(1, valid.sum() - 2)
        sigma2 = np.sum(resid ** 2) / dof
        sigma = np.sqrt(sigma2)
        # simple prediction SE (approx): sigma
        pred_se[i] = sigma

    # Determine integer layer count to present (prefer max layers among source models)
    max_layers = int(max(m['layers'][-1] for m in models)) + 1
    if max_layers < 30:
        max_layers = 36

    layers_int = np.arange(0, max_layers)
    rel_layers = layers_int / float(max_layers - 1)

    # Interpolate predicted values onto integer-layer relative positions
    preds_layers = np.interp(rel_layers, rel_grid, preds, left=np.nan, right=np.nan)

    # Save CSV with one row per integer layer
    out_csv = results / 'twonn_14B_estimated.csv'
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['layer', 'est_intrinsic_dim'])
        for layer, val in zip(layers_int, preds_layers):
            w.writerow([int(layer), float(val) if not np.isnan(val) else ''])

    # Plot one point per layer (marker + line) like example
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(layers_int, preds_layers, marker='o', markersize=5, linewidth=1.25, color='C0')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Estimated intrinsic dimension (Two-NN)')
    ax.set_title('Qwen/Qwen3-14B - Two-NN intrinsic dimension vs depth')
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()
    out_png = results / 'twonn_14B_estimated.png'
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    print(f'Wrote {out_csv} and {out_png}')


if __name__ == '__main__':
    main()

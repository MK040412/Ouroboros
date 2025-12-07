#!/usr/bin/env python3
"""
Compare 3 training runs:
1. COYO (longest run from data/loss.csv)
2. ImageNet Previous (buggy embeddings)
3. ImageNet Current (fixed embeddings)
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path


def load_and_split_runs(path):
    """Load CSV and split into separate runs based on epoch 1, step 100."""
    runs = []
    current_run = []

    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = int(row['epoch'])
            step = int(row['step'])

            if epoch == 1 and step == 100:
                if current_run:
                    runs.append(current_run)
                current_run = []

            current_run.append({
                'timestamp': row['timestamp'],
                'epoch': epoch,
                'step': step,
                'global_step': int(row['global_step']),
                'loss': float(row['loss']),
            })

    if current_run:
        runs.append(current_run)

    return runs


def power_law(x, a, b, c):
    """Power law: y = a * x^b + c"""
    return a * np.power(x, b) + c


def fit_power_law(steps, losses, bounds=None):
    """Fit power law to data."""
    try:
        if bounds is None:
            bounds = ([0.1, -2, 0], [100, 0, 5])
        popt, _ = curve_fit(
            power_law, steps, losses,
            p0=[2.0, -0.4, 0.75],
            bounds=bounds,
            maxfev=10000
        )
        y_pred = power_law(steps, *popt)
        ss_res = np.sum((losses - y_pred) ** 2)
        ss_tot = np.sum((losses - np.mean(losses)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        return popt, r_squared
    except Exception as e:
        print(f"Fitting failed: {e}")
        return None, None


def main():
    output_dir = Path('./outputs/loss')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load ImageNet runs
    imagenet_runs = load_and_split_runs('loss_imagenet.csv')
    imagenet_sorted = sorted(enumerate(imagenet_runs), key=lambda x: len(x[1]), reverse=True)

    # Get two longest ImageNet runs
    prev_idx, prev_run = imagenet_sorted[0]
    curr_idx, curr_run = imagenet_sorted[1]
    if prev_run[0]['timestamp'] > curr_run[0]['timestamp']:
        prev_idx, prev_run, curr_idx, curr_run = curr_idx, curr_run, prev_idx, prev_run

    # Load COYO runs and get the one with loss in 0~1 range (Run 5)
    coyo_runs = load_and_split_runs('data/loss.csv')
    # Run 5 (index 4) has loss in 0~1 range
    coyo_idx = 4
    coyo_run = coyo_runs[coyo_idx] if len(coyo_runs) > coyo_idx else coyo_runs[-1]

    print(f"Comparing 3 runs:")
    print(f"  1. COYO (Run {coyo_idx+1}): {len(coyo_run)} points")
    print(f"  2. ImageNet Previous (buggy): {len(prev_run)} points")
    print(f"  3. ImageNet Current (fixed): {len(curr_run)} points")

    # Extract and filter data
    def extract_data(run, max_loss=1.0):
        steps = np.array([d['global_step'] for d in run])
        losses = np.array([d['loss'] for d in run])
        mask = losses < max_loss
        return steps[mask], losses[mask]

    coyo_steps, coyo_losses = extract_data(coyo_run, max_loss=1.0)  # Filter to 0~1 range
    prev_steps, prev_losses = extract_data(prev_run, max_loss=1.0)
    curr_steps, curr_losses = extract_data(curr_run, max_loss=1.0)

    print(f"\nFiltered data points:")
    print(f"  COYO: {len(coyo_steps)}")
    print(f"  ImageNet Previous: {len(prev_steps)}")
    print(f"  ImageNet Current: {len(curr_steps)}")

    # Fit power law (step >= 200)
    fit_mask_coyo = coyo_steps >= 200
    fit_mask_prev = prev_steps >= 200
    fit_mask_curr = curr_steps >= 200

    coyo_params, coyo_r2 = fit_power_law(coyo_steps[fit_mask_coyo], coyo_losses[fit_mask_coyo],
                                          bounds=([0.1, -2, 0.5], [100, 0, 2]))
    prev_params, prev_r2 = fit_power_law(prev_steps[fit_mask_prev], prev_losses[fit_mask_prev])
    curr_params, curr_r2 = fit_power_law(curr_steps[fit_mask_curr], curr_losses[fit_mask_curr])

    print(f"\nPower law fits (y = a*x^b + c):")
    if coyo_params is not None:
        print(f"  COYO: {coyo_params[0]:.3f} * x^({coyo_params[1]:.3f}) + {coyo_params[2]:.3f} (R²={coyo_r2:.3f})")
    if prev_params is not None:
        print(f"  ImageNet Prev: {prev_params[0]:.3f} * x^({prev_params[1]:.3f}) + {prev_params[2]:.3f} (R²={prev_r2:.3f})")
    if curr_params is not None:
        print(f"  ImageNet Curr: {curr_params[0]:.3f} * x^({curr_params[1]:.3f}) + {curr_params[2]:.3f} (R²={curr_r2:.3f})")

    # ========================================
    # Plot 1: All three curves comparison
    # ========================================
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot data
    ax.plot(coyo_steps, coyo_losses, 'g-', linewidth=0.8, alpha=0.7, label='COYO')
    ax.plot(prev_steps, prev_losses, 'b-', linewidth=0.8, alpha=0.7, label='ImageNet (buggy)')
    ax.plot(curr_steps, curr_losses, 'r-', linewidth=0.8, alpha=0.8, label='ImageNet (fixed)')

    # Plot fitted curves
    colors = ['g', 'b', 'r']
    params_list = [coyo_params, prev_params, curr_params]
    steps_list = [coyo_steps, prev_steps, curr_steps]
    names = ['COYO', 'IN-buggy', 'IN-fixed']

    for params, steps, color, name in zip(params_list, steps_list, colors, names):
        if params is not None:
            x_fit = np.logspace(np.log10(200), np.log10(steps.max()), 200)
            y_fit = power_law(x_fit, *params)
            ax.plot(x_fit, y_fit, f'{color}--', linewidth=2, alpha=0.8,
                    label=f'{name}: {params[0]:.2f}·x^({params[1]:.3f})+{params[2]:.3f}')

    ax.set_xscale('log')
    ax.set_xlabel('Global Step (log scale)', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Comparison: COYO vs ImageNet (buggy vs fixed)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    output_path = output_dir / 'all_runs_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_path}")

    # ========================================
    # Plot 2: Separate panels with power law fit
    # ========================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    data_list = [
        (coyo_steps, coyo_losses, coyo_params, 'g', 'COYO'),
        (prev_steps, prev_losses, prev_params, 'b', 'ImageNet (buggy)'),
        (curr_steps, curr_losses, curr_params, 'r', 'ImageNet (fixed)'),
    ]

    for ax, (steps, losses, params, color, name) in zip(axes, data_list):
        fit_mask = steps >= 200
        ax.scatter(steps[fit_mask], losses[fit_mask], c=color, alpha=0.5, s=10, label='Data')

        if params is not None:
            a, b, c = params
            x_line = np.logspace(np.log10(steps[fit_mask].min()), np.log10(steps[fit_mask].max()), 200)
            y_line = a * np.power(x_line, b) + c
            ax.plot(x_line, y_line, f'{color}--', linewidth=2,
                    label=f'y = {a:.2f}·x^({b:.3f}) + {c:.3f}')
            ax.axhline(y=c, color=color, linestyle=':', alpha=0.5, label=f'asymptote: {c:.4f}')

        ax.set_xscale('log')
        ax.set_xlabel('Step (log scale)', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'{name}: b={params[1]:.3f}' if params is not None else name, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    output_path = output_dir / 'all_runs_separate.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

    # ========================================
    # Plot 3: Linear fit in multiple regions for ImageNet only
    # ========================================

    # Define regions to analyze
    regions = [
        (6e3, 1e4, '6×10³~10⁴'),
        (1e4, 3e4, '10⁴~3×10⁴'),
        (3e4, 1e5, '3×10⁴~10⁵'),
        (6e3, 1e5, '6×10³~10⁵ (full)'),
    ]

    # Only ImageNet data
    imagenet_data = [
        (prev_steps, prev_losses, 'b', 'ImageNet (buggy)'),
        (curr_steps, curr_losses, 'r', 'ImageNet (fixed)'),
    ]

    print(f"\n=== Linear Fit Results (y = slope·log₁₀(x) + intercept) ===")
    print(f"{'Region':<20} {'Dataset':<20} {'Slope':<12} {'Intercept':<12}")
    print("-" * 64)

    results = []
    for start, end, region_name in regions:
        for steps, losses, color, name in imagenet_data:
            region_mask = (steps >= start) & (steps <= end)
            x_data = steps[region_mask]
            y_data = losses[region_mask]

            if len(x_data) > 10:
                log_x = np.log10(x_data)
                coeffs = np.polyfit(log_x, y_data, 1)
                slope, intercept = coeffs
                results.append((region_name, name, slope, intercept, x_data, y_data, color))
                print(f"{region_name:<20} {name:<20} {slope:<12.4f} {intercept:<12.4f}")

    # Plot: 6×10³ ~ 10⁵ full range comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, (steps, losses, color, name) in zip(axes, imagenet_data):
        region_mask = (steps >= 6e3) & (steps <= 1e5)
        x_data = steps[region_mask]
        y_data = losses[region_mask]

        if len(x_data) > 10:
            log_x = np.log10(x_data)
            coeffs = np.polyfit(log_x, y_data, 1)
            slope, intercept = coeffs

            ax.scatter(x_data, y_data, c=color, alpha=0.5, s=10, label='Data')

            x_line = np.logspace(np.log10(6e3), np.log10(x_data.max()), 100)
            y_line = slope * np.log10(x_line) + intercept
            ax.plot(x_line, y_line, f'{color}--', linewidth=2,
                    label=f'y = {slope:.4f}·log₁₀(x) + {intercept:.3f}')

            ax.set_xscale('log')
            ax.set_xlabel('Step (log scale)', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title(f'{name}: 6×10³~10⁵, slope={slope:.4f}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    output_path = output_dir / 'imagenet_linear_full_region.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_path}")

    # Plot: Multiple regions comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (start, end, region_name) in zip(axes, regions):
        for steps, losses, color, name in imagenet_data:
            region_mask = (steps >= start) & (steps <= end)
            x_data = steps[region_mask]
            y_data = losses[region_mask]

            if len(x_data) > 5:
                log_x = np.log10(x_data)
                coeffs = np.polyfit(log_x, y_data, 1)
                slope, intercept = coeffs

                ax.scatter(x_data, y_data, c=color, alpha=0.4, s=8, label=f'{name}')

                x_line = np.logspace(np.log10(x_data.min()), np.log10(x_data.max()), 100)
                y_line = slope * np.log10(x_line) + intercept
                ax.plot(x_line, y_line, f'{color}--', linewidth=2,
                        label=f'slope={slope:.4f}')

        ax.set_xscale('log')
        ax.set_xlabel('Step (log scale)', fontsize=10)
        ax.set_ylabel('Loss', fontsize=10)
        ax.set_title(f'Region: {region_name}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    output_path = output_dir / 'imagenet_linear_regions.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

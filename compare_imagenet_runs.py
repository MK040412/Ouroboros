#!/usr/bin/env python3
"""
Compare two ImageNet training runs with power law fitting.
Separates runs by finding epoch 1, step 100 entries.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
from datetime import datetime


def load_and_split_runs(path):
    """Load CSV and split into separate runs based on epoch 1, step 100."""
    runs = []
    current_run = []

    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = int(row['epoch'])
            step = int(row['step'])

            # New run starts at epoch 1, step 100
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


def fit_power_law(steps, losses):
    """Fit power law to data, return (a, b, c) and R²."""
    try:
        # Initial guess
        popt, _ = curve_fit(
            power_law, steps, losses,
            p0=[2.0, -0.4, 0.75],
            bounds=([0.1, -2, 0], [10, 0, 2]),
            maxfev=10000
        )

        # Calculate R²
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

    # Load and split runs
    runs = load_and_split_runs('loss_imagenet.csv')

    print(f"Found {len(runs)} training runs:")
    for i, run in enumerate(runs):
        print(f"  Run {i+1}: {len(run)} points, started {run[0]['timestamp'][:19]}")

    # Get the two longest runs
    runs_sorted = sorted(enumerate(runs), key=lambda x: len(x[1]), reverse=True)

    if len(runs_sorted) < 2:
        print("Need at least 2 runs to compare!")
        return

    prev_idx, prev_run = runs_sorted[0]
    curr_idx, curr_run = runs_sorted[1]

    # Make sure "current" is the more recent one
    if prev_run[0]['timestamp'] > curr_run[0]['timestamp']:
        prev_idx, prev_run, curr_idx, curr_run = curr_idx, curr_run, prev_idx, prev_run

    print(f"\nComparing:")
    print(f"  Previous (Run {prev_idx+1}): {len(prev_run)} points - buggy embeddings")
    print(f"  Current (Run {curr_idx+1}): {len(curr_run)} points - fixed embeddings")

    # Extract data and filter spikes
    prev_steps = np.array([d['global_step'] for d in prev_run])
    prev_losses = np.array([d['loss'] for d in prev_run])
    curr_steps = np.array([d['global_step'] for d in curr_run])
    curr_losses = np.array([d['loss'] for d in curr_run])

    # Filter outliers (loss < 1.0)
    prev_mask = prev_losses < 1.0
    curr_mask = curr_losses < 1.0

    prev_steps_f = prev_steps[prev_mask]
    prev_losses_f = prev_losses[prev_mask]
    curr_steps_f = curr_steps[curr_mask]
    curr_losses_f = curr_losses[curr_mask]

    # Start from step >= 200 for fitting (skip warmup)
    prev_fit_mask = prev_steps_f >= 200
    curr_fit_mask = curr_steps_f >= 200

    print(f"\nData points for fitting:")
    print(f"  Previous: {prev_fit_mask.sum()} (filtered {(~prev_mask).sum()} spikes)")
    print(f"  Current: {curr_fit_mask.sum()} (filtered {(~curr_mask).sum()} spikes)")

    # Fit power law
    prev_params, prev_r2 = fit_power_law(prev_steps_f[prev_fit_mask], prev_losses_f[prev_fit_mask])
    curr_params, curr_r2 = fit_power_law(curr_steps_f[curr_fit_mask], curr_losses_f[curr_fit_mask])

    if prev_params is not None:
        print(f"\nPrevious fit: {prev_params[0]:.3f} * x^({prev_params[1]:.3f}) + {prev_params[2]:.3f} (R²={prev_r2:.3f})")
    if curr_params is not None:
        print(f"Current fit: {curr_params[0]:.3f} * x^({curr_params[1]:.3f}) + {curr_params[2]:.3f} (R²={curr_r2:.3f})")

    # ========================================
    # Plot 1: Log scale X, linear Y with power law fit
    # ========================================
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot data
    ax.plot(prev_steps_f, prev_losses_f, 'b-', linewidth=0.8, alpha=0.7, label='Previous (buggy embed)')
    ax.plot(curr_steps_f, curr_losses_f, 'r-', linewidth=0.8, alpha=0.8, label='Current (fixed embed)')

    # Plot fitted curves
    if prev_params is not None:
        x_fit = np.logspace(np.log10(200), np.log10(prev_steps_f.max()), 200)
        y_fit = power_law(x_fit, *prev_params)
        ax.plot(x_fit, y_fit, 'b--', linewidth=2, alpha=0.8,
                label=f'Prev fit: {prev_params[0]:.2f}*x^({prev_params[1]:.3f})+{prev_params[2]:.3f} (R²={prev_r2:.3f})')
        ax.axhline(y=prev_params[2], color='b', linestyle=':', alpha=0.5, label=f'Prev asymptote: {prev_params[2]:.4f}')

    if curr_params is not None:
        x_fit = np.logspace(np.log10(200), np.log10(curr_steps_f.max()), 200)
        y_fit = power_law(x_fit, *curr_params)
        ax.plot(x_fit, y_fit, 'r--', linewidth=2, alpha=0.8,
                label=f'Curr fit: {curr_params[0]:.2f}*x^({curr_params[1]:.3f})+{curr_params[2]:.3f} (R²={curr_r2:.3f})')
        ax.axhline(y=curr_params[2], color='r', linestyle=':', alpha=0.5, label=f'Curr asymptote: {curr_params[2]:.4f}')

    ax.set_xscale('log')
    ax.set_xlabel('Global Step (log scale)', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_ylim(0.7, 1.1)
    ax.set_title('ImageNet Training Comparison - Power Law Fit', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    output_path = output_dir / 'imagenet_runs_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_path}")

    # ========================================
    # Plot 2: Linearized view - x: log(step), y: loss
    # y = a*x^b + c  형태로, x축을 log scale로 표시
    # ========================================
    if prev_params is not None and curr_params is not None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Left: Previous
        ax = axes[0]
        a_prev, b_prev, c_prev = prev_params
        x_data = prev_steps_f[prev_fit_mask]
        y_data = prev_losses_f[prev_fit_mask]

        ax.scatter(x_data, y_data, c='blue', alpha=0.5, s=10, label='Data')

        # Fit curve
        x_line = np.logspace(np.log10(x_data.min()), np.log10(x_data.max()), 200)
        y_line = a_prev * np.power(x_line, b_prev) + c_prev
        ax.plot(x_line, y_line, 'b--', linewidth=2,
                label=f'y = {a_prev:.2f}·x^({b_prev:.3f}) + {c_prev:.3f}')
        ax.axhline(y=c_prev, color='b', linestyle=':', alpha=0.5, label=f'asymptote: {c_prev:.4f}')

        ax.set_xscale('log')
        ax.set_xlabel('Step (log scale)', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'Previous (buggy): b={b_prev:.3f}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')

        # Right: Current
        ax = axes[1]
        a_curr, b_curr, c_curr = curr_params
        x_data = curr_steps_f[curr_fit_mask]
        y_data = curr_losses_f[curr_fit_mask]

        ax.scatter(x_data, y_data, c='red', alpha=0.5, s=10, label='Data')

        x_line = np.logspace(np.log10(x_data.min()), np.log10(x_data.max()), 200)
        y_line = a_curr * np.power(x_line, b_curr) + c_curr
        ax.plot(x_line, y_line, 'r--', linewidth=2,
                label=f'y = {a_curr:.2f}·x^({b_curr:.3f}) + {c_curr:.3f}')
        ax.axhline(y=c_curr, color='r', linestyle=':', alpha=0.5, label=f'asymptote: {c_curr:.4f}')

        ax.set_xscale('log')
        ax.set_xlabel('Step (log scale)', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'Current (fixed): b={b_curr:.3f}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')

        plt.tight_layout()
        output_path = output_dir / 'imagenet_runs_linearized.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")

    # ========================================
    # Plot 3: Linear fit in 10^4 ~ 10^5 region
    # y = -ax + b  where x = log10(step), y = loss
    # ========================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Previous (10^4 ~ 10^5)
    ax = axes[0]
    region_mask = (prev_steps_f >= 1e4) & (prev_steps_f <= 1e5)
    x_data = prev_steps_f[region_mask]
    y_data = prev_losses_f[region_mask]

    if len(x_data) > 10:
        log_x = np.log10(x_data)
        # Linear fit: y = slope * log_x + intercept
        coeffs = np.polyfit(log_x, y_data, 1)
        slope, intercept = coeffs

        ax.scatter(x_data, y_data, c='blue', alpha=0.5, s=10, label='Data')

        x_line = np.logspace(4, 5, 100)
        y_line = slope * np.log10(x_line) + intercept
        ax.plot(x_line, y_line, 'b--', linewidth=2,
                label=f'y = {slope:.4f}·log₁₀(x) + {intercept:.3f}')

        ax.set_xscale('log')
        ax.set_xlabel('Step (log scale)', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'Previous (buggy): 10⁴~10⁵ region, slope={slope:.4f}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')

        print(f"\nPrevious (10^4 ~ 10^5): y = {slope:.4f} * log10(x) + {intercept:.3f}")
    else:
        ax.text(0.5, 0.5, 'Not enough data in range', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Previous (buggy): 10⁴~10⁵ region')

    # Right: Current (10^4 ~ 10^5)
    ax = axes[1]
    region_mask = (curr_steps_f >= 1e4) & (curr_steps_f <= 1e5)
    x_data = curr_steps_f[region_mask]
    y_data = curr_losses_f[region_mask]

    if len(x_data) > 10:
        log_x = np.log10(x_data)
        coeffs = np.polyfit(log_x, y_data, 1)
        slope, intercept = coeffs

        ax.scatter(x_data, y_data, c='red', alpha=0.5, s=10, label='Data')

        x_line = np.logspace(4, np.log10(x_data.max()), 100)
        y_line = slope * np.log10(x_line) + intercept
        ax.plot(x_line, y_line, 'r--', linewidth=2,
                label=f'y = {slope:.4f}·log₁₀(x) + {intercept:.3f}')

        ax.set_xscale('log')
        ax.set_xlabel('Step (log scale)', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'Current (fixed): 10⁴~10⁵ region, slope={slope:.4f}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')

        print(f"Current (10^4 ~ 10^5): y = {slope:.4f} * log10(x) + {intercept:.3f}")
    else:
        ax.text(0.5, 0.5, 'Not enough data in range', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Current (fixed): 10⁴~10⁵ region')

    plt.tight_layout()
    output_path = output_dir / 'imagenet_runs_linear_region.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

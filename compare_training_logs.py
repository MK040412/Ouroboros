#!/usr/bin/env python3
"""
Compare current and previous ImageNet training logs.
Shows log-log plot with both curves overlaid.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_csv(path):
    data = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'epoch': int(row['epoch']),
                'global_step': int(row['global_step']),
                'loss': float(row['loss']),
            })
    return data


def main():
    output_dir = Path('./outputs/loss')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    current = load_csv('loss_imagenet.csv')
    previous = load_csv('data/loss_imagenet.csv')

    curr_step = np.array([d['global_step'] for d in current])
    curr_loss = np.array([d['loss'] for d in current])
    prev_step = np.array([d['global_step'] for d in previous])
    prev_loss = np.array([d['loss'] for d in previous])

    # Filter spikes
    curr_mask = curr_loss < 10
    prev_mask = prev_loss < 10

    curr_step_f = curr_step[curr_mask]
    curr_loss_f = curr_loss[curr_mask]
    prev_step_f = prev_step[prev_mask]
    prev_loss_f = prev_loss[prev_mask]

    print(f"Current: {len(curr_step_f)} points (filtered {(~curr_mask).sum()} spikes)")
    print(f"Previous: {len(prev_step_f)} points (filtered {(~prev_mask).sum()} spikes)")

    # ========================================
    # Plot 1: Full comparison (log-linear, y: 0.7-1.0)
    # ========================================
    fig, ax = plt.subplots(figsize=(14, 7))

    # Filter to y range [0.7, 1.0]
    prev_y_mask = (prev_loss_f >= 0.7) & (prev_loss_f <= 1.0)
    curr_y_mask = (curr_loss_f >= 0.7) & (curr_loss_f <= 1.0)

    ax.plot(prev_step_f[prev_y_mask], prev_loss_f[prev_y_mask], 'b-', linewidth=0.8, alpha=0.6,
            label=f'Previous (151 epochs, buggy embeddings)')
    ax.plot(curr_step_f[curr_y_mask], curr_loss_f[curr_y_mask], 'r-', linewidth=0.8, alpha=0.8,
            label=f'Current (fixed embeddings)')

    ax.set_xscale('log')
    ax.set_xlabel('Global Step (log scale)', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_ylim(0.7, 1.0)
    ax.set_title('ImageNet Training Comparison - Log Scale', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper right')

    plt.tight_layout()
    output_path = output_dir / 'loss_comparison_full.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

    # ========================================
    # Plot 2: Overlapping region only (step 0 ~ 30000)
    # ========================================
    max_step = min(curr_step_f.max(), 35000)

    curr_overlap_mask = curr_step_f <= max_step
    prev_overlap_mask = prev_step_f <= max_step

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(prev_step_f[prev_overlap_mask], prev_loss_f[prev_overlap_mask],
            'b-', linewidth=1.0, alpha=0.7, label='Previous (buggy embeddings)')
    ax.plot(curr_step_f[curr_overlap_mask], curr_loss_f[curr_overlap_mask],
            'r-', linewidth=1.0, alpha=0.8, label='Current (fixed embeddings)')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Global Step (log scale)', fontsize=12)
    ax.set_ylabel('Loss (log scale)', fontsize=12)
    ax.set_title(f'Training Comparison - Overlapping Region (step ≤ {max_step:,})',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper right')

    plt.tight_layout()
    output_path = output_dir / 'loss_comparison_overlap.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

    # ========================================
    # Plot 3: Three segments comparison
    # ========================================
    segments = [
        (0, 1000, 'Early (0-1K steps)'),
        (1000, 10000, 'Mid (1K-10K steps)'),
        (10000, 35000, 'Late (10K-35K steps)'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (start, end, title) in zip(axes, segments):
        curr_seg_mask = (curr_step_f >= start) & (curr_step_f <= end)
        prev_seg_mask = (prev_step_f >= start) & (prev_step_f <= end)

        if prev_seg_mask.sum() > 0:
            ax.plot(prev_step_f[prev_seg_mask], prev_loss_f[prev_seg_mask],
                    'b-', linewidth=1.0, alpha=0.7, label='Previous')
        if curr_seg_mask.sum() > 0:
            ax.plot(curr_step_f[curr_seg_mask], curr_loss_f[curr_seg_mask],
                    'r-', linewidth=1.0, alpha=0.8, label='Current')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Step', fontsize=10)
        ax.set_ylabel('Loss', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    output_path = output_dir / 'loss_comparison_segments.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

    # ========================================
    # Statistics comparison at same steps
    # ========================================
    print("\n=== Loss Comparison at Key Steps ===")

    key_steps = [1000, 5000, 10000, 20000, 30000]

    for step in key_steps:
        # Find closest step in each dataset
        curr_idx = np.argmin(np.abs(curr_step - step))
        prev_idx = np.argmin(np.abs(prev_step - step))

        curr_val = curr_loss[curr_idx] if curr_loss[curr_idx] < 10 else float('nan')
        prev_val = prev_loss[prev_idx] if prev_loss[prev_idx] < 10 else float('nan')

        diff = curr_val - prev_val if not (np.isnan(curr_val) or np.isnan(prev_val)) else float('nan')

        print(f"  Step {step:>6}: Current={curr_val:.4f}, Previous={prev_val:.4f}, Diff={diff:+.4f}")


if __name__ == "__main__":
    main()

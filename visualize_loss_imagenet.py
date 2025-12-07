#!/usr/bin/env python3
"""
Visualize ImageNet training loss from CSV log file.

Usage:
    python visualize_loss_imagenet.py                    # Full data (from line 2)
    python visualize_loss_imagenet.py --from 100         # Start from raw line 100
    python visualize_loss_imagenet.py --smoothing 20     # Add smoothed line
"""

import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Visualize ImageNet training loss')
    parser.add_argument('--input', '-i', type=str, default='loss_imagenet.csv',
                        help='Input CSV file path')
    parser.add_argument('--output-dir', '-o', type=str, default='./outputs/loss',
                        help='Output directory for images')
    parser.add_argument('--from', '-n', type=int, default=2, dest='from_row',
                        help='Start from raw line N (1=header, 2=first data row)')
    parser.add_argument('--smoothing', '-s', type=int, default=None,
                        help='Smoothing window size for moving average')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read CSV
    data = []
    with open(args.input, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'epoch': int(row['epoch']),
                'step': int(row['step']),
                'global_step': int(row['global_step']),
                'loss': float(row['loss']),
                'lr': float(row['lr']),
            })

    print(f"Loaded {len(data)} rows from {args.input}")

    # Apply --from filter (raw line number: 1=header, 2=first data)
    skip_count = args.from_row - 2  # line 2 = first data = index 0
    if skip_count > 0:
        data = data[skip_count:]
        print(f"Starting from raw line {args.from_row}, {len(data)} rows remaining")

    if len(data) == 0:
        print("No data to plot!")
        return

    # Extract arrays
    global_step = np.array([d['global_step'] for d in data])
    loss = np.array([d['loss'] for d in data])
    epoch = np.array([d['epoch'] for d in data])
    step = np.array([d['step'] for d in data])
    lr = np.array([d['lr'] for d in data])

    print(f"Data range: step {global_step[0]} to {global_step[-1]}")
    print(f"Loss range: {loss.min():.6f} to {loss.max():.6f}")
    print(f"Epochs: {epoch.min()} to {epoch.max()}")

    # Find epoch change points
    epoch_changes_idx = np.where(np.diff(epoch) != 0)[0] + 1
    epoch_changes_idx = np.concatenate([[0], epoch_changes_idx])
    epoch_change_steps = global_step[epoch_changes_idx]
    epoch_change_epochs = epoch[epoch_changes_idx]

    # ========================================
    # Plot 1: Linear scale (x: linear, y: linear)
    # ========================================
    fig, ax1 = plt.subplots(figsize=(14, 7))

    ax1.plot(global_step, loss, 'b-', linewidth=0.8, alpha=0.6, label='Loss')

    # Smoothed line
    if args.smoothing and len(loss) >= args.smoothing:
        kernel = np.ones(args.smoothing) / args.smoothing
        smoothed = np.convolve(loss, kernel, mode='valid')
        smoothed_steps = global_step[args.smoothing-1:]
        ax1.plot(smoothed_steps, smoothed, 'r-', linewidth=2,
                 label=f'Smoothed (MA-{args.smoothing})')

    ax1.set_xlabel('Global Step', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    # Y-axis: fixed upper limit to clip spikes
    min_y = loss.min()
    max_y = 1.0
    y_padding = (max_y - min_y) * 0.05
    ax1.set_ylim(min_y - y_padding, max_y + y_padding)
    ax1.set_title('ImageNet Training Loss (XUT-Small)', fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')

    # Secondary x-axis: epoch markers
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())

    # Show epoch markers (every N epochs to avoid clutter)
    n_epochs = len(epoch_change_epochs)
    if n_epochs <= 20:
        step_size = 1
    elif n_epochs <= 50:
        step_size = 5
    else:
        step_size = 10

    tick_positions = epoch_change_steps[::step_size]
    tick_labels = [f"E{int(e)}" for e in epoch_change_epochs[::step_size]]
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, fontsize=9)
    ax2.set_xlabel('Epoch', fontsize=12)

    plt.tight_layout()
    output_path = output_dir / 'loss_imagenet.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

    # ========================================
    # Plot 2: Log scale (x: log, y: linear)
    # ========================================
    fig, ax1 = plt.subplots(figsize=(14, 7))

    ax1.plot(global_step, loss, 'b-', linewidth=0.8, alpha=0.6, label='Loss')

    # Smoothed line
    if args.smoothing and len(loss) >= args.smoothing:
        ax1.plot(smoothed_steps, smoothed, 'r-', linewidth=2,
                 label=f'Smoothed (MA-{args.smoothing})')

    ax1.set_xscale('log')
    ax1.set_xlabel('Global Step (log scale)', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_ylim(min_y - y_padding, max_y + y_padding)
    ax1.set_title('ImageNet Training Loss - Log Scale (XUT-Small)', fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(loc='upper right')

    # Secondary x-axis: epoch markers (log scale)
    ax2 = ax1.twiny()
    ax2.set_xscale('log')
    ax2.set_xlim(ax1.get_xlim())

    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, fontsize=9)
    ax2.set_xlabel('Epoch', fontsize=12)

    plt.tight_layout()
    output_path = output_dir / 'loss_imagenet_logscale.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

    # Print summary statistics
    print(f"\n[Summary]")
    print(f"  Total steps: {global_step[-1] - global_step[0]:,}")
    print(f"  Total epochs: {epoch.max()}")
    print(f"  Final loss: {loss[-1]:.6f}")
    print(f"  Min loss: {loss.min():.6f} at step {global_step[loss.argmin()]:,}")
    if len(loss) >= 100:
        print(f"  Mean loss (last 100): {loss[-100:].mean():.6f}")
    else:
        print(f"  Mean loss: {loss.mean():.6f}")


if __name__ == "__main__":
    main()

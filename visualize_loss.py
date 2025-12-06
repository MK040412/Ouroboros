#!/usr/bin/env python3
"""
Loss Visualization Script

Reads loss.csv from TPU training and generates loss.png plot.
- Automatically detects and uses the most recent training session
- Dual x-axis: epoch (top) and step (bottom)
- Log scale for loss

Usage:
    python visualize_loss.py                           # Use default ./loss.csv
    python visualize_loss.py --input path/to/loss.csv  # Custom input path
    python visualize_loss.py --output custom_loss.png  # Custom output path
    python visualize_loss.py --all                     # Show all sessions, not just latest
"""

import argparse
from pathlib import Path
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np


def load_loss_data(csv_path: str) -> pd.DataFrame:
    """Load loss data from CSV file"""
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"Loaded {len(df)} records from {csv_path}")
    return df


def detect_sessions(df: pd.DataFrame, gap_threshold_minutes: int = 30) -> pd.DataFrame:
    """Detect training sessions based on timestamp gaps

    A new session starts when there's a gap > gap_threshold_minutes
    or when global_step resets (goes backward)
    """
    df = df.copy()
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Detect gaps
    time_diff = df['timestamp'].diff()
    gap_threshold = timedelta(minutes=gap_threshold_minutes)

    # Detect step resets (global_step goes backward)
    step_diff = df['global_step'].diff()

    # New session: large time gap OR step reset
    new_session = (time_diff > gap_threshold) | (step_diff < 0)
    df['session'] = new_session.cumsum()

    # Count sessions
    n_sessions = df['session'].nunique()
    print(f"Detected {n_sessions} training session(s)")

    for sid in df['session'].unique():
        session_df = df[df['session'] == sid]
        start_time = session_df['timestamp'].min()
        end_time = session_df['timestamp'].max()
        steps = len(session_df)
        step_range = f"{session_df['global_step'].min()}-{session_df['global_step'].max()}"
        print(f"  Session {sid}: {start_time} ~ {end_time} ({steps} records, steps {step_range})")

    return df


def get_latest_session(df: pd.DataFrame) -> pd.DataFrame:
    """Extract only the latest training session"""
    latest_session_id = df['session'].max()
    latest_df = df[df['session'] == latest_session_id].copy()
    print(f"\nUsing latest session (session {latest_session_id}): {len(latest_df)} records")
    return latest_df


def fit_power_law(steps: np.ndarray, loss: np.ndarray):
    """Fit power law model: loss = a * step^(-b) + c

    Uses data from 10% position onwards to skip initial unstable region.
    Fits in log-log space: log(loss - c) = log(a) - b * log(step)

    Returns:
        tuple: (a, b, c, r_squared) coefficients and R² score
    """
    from scipy.optimize import curve_fit

    # Use data from 10% position onwards (skip initial unstable region)
    start_idx = len(steps) // 10
    fit_steps = steps[start_idx:]
    fit_loss = loss[start_idx:]

    # Filter out any spikes (loss > 2)
    mask = fit_loss < 2
    fit_steps = fit_steps[mask]
    fit_loss = fit_loss[mask]

    if len(fit_steps) < 10:
        return None, None, None, None

    # Power law with asymptote: loss = a * step^(-b) + c
    def power_law(x, a, b, c):
        return a * np.power(x, -b) + c

    try:
        # Initial guess: a=1, b=0.5, c=min_loss
        p0 = [1.0, 0.5, fit_loss.min() * 0.9]
        bounds = ([0, 0, 0], [np.inf, 2, fit_loss.min()])

        popt, _ = curve_fit(power_law, fit_steps, fit_loss, p0=p0, bounds=bounds, maxfev=5000)
        a, b, c = popt

        # R² score
        y_pred = power_law(fit_steps, a, b, c)
        ss_res = np.sum((fit_loss - y_pred) ** 2)
        ss_tot = np.sum((fit_loss - np.mean(fit_loss)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return a, b, c, r_squared

    except Exception as e:
        print(f"  [Warning] Power law fitting failed: {e}")
        return None, None, None, None


def predict_steps_for_loss_power(target_loss: float, a: float, b: float, c: float) -> float:
    """Predict step count to reach target loss using loss = a * step^(-b) + c

    Solving for step: step = ((target_loss - c) / a)^(-1/b)
    """
    if target_loss <= c:
        # Target is below asymptote - unreachable
        return float('inf')

    if a <= 0 or b <= 0:
        return float('inf')

    try:
        step = np.power((target_loss - c) / a, -1.0 / b)
        return step
    except:
        return float('inf')


def print_eta_predictions(df: pd.DataFrame, current_step: int, steps_per_second: float = None):
    """Print ETA predictions for various loss targets using power law model"""

    steps = df['global_step'].values
    loss = df['loss'].values

    # Fit model
    a, b, c, r_squared = fit_power_law(steps, loss)

    if a is None:
        print("\n[ETA] Insufficient data for fitting")
        return

    print("\n" + "=" * 60)
    print("Loss Prediction (model: loss = a * step^(-b) + c)")
    print("=" * 60)
    print(f"  Fitted: loss = {a:.4f} * step^(-{b:.4f}) + {c:.4f}")
    print(f"  Asymptote (c): {c:.4f} (theoretical minimum loss)")
    print(f"  R² = {r_squared:.4f}")

    if r_squared < 0.5:
        print("  [WARNING] Poor fit (R² < 0.5), predictions may be unreliable")

    # Estimate steps per second if not provided
    if steps_per_second is None:
        time_diff = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds()
        step_diff = steps[-1] - steps[0]
        steps_per_second = step_diff / time_diff if time_diff > 0 else 0

    print(f"  Current step: {current_step:,}")
    print(f"  Speed: {steps_per_second:.2f} steps/sec")

    current_loss = loss[-1]
    print(f"  Current loss: {current_loss:.4f}")

    # Predict for target losses
    targets = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    targets = [t for t in targets if t < current_loss]  # Only show targets below current

    if not targets:
        print("\n  No targets below current loss")
        return

    print(f"\n  {'Target':<10} {'Steps':>15} {'Remaining':>15} {'ETA':>20}")
    print("  " + "-" * 60)

    for target in targets:
        if target <= c:
            # Below asymptote
            print(f"  loss={target:<5.1f} {'N/A':>15} {'N/A':>15} {'Below asymptote':>20}")
            continue

        predicted_step = predict_steps_for_loss_power(target, a, b, c)
        remaining_steps = predicted_step - current_step

        if remaining_steps <= 0:
            eta_str = "Already reached"
            step_str = "-"
            remaining_str = "-"
        elif predicted_step > 1e15:
            eta_str = "Unreachable"
            step_str = ">1e15"
            remaining_str = "-"
        else:
            step_str = f"{predicted_step:,.0f}"
            remaining_str = f"{remaining_steps:,.0f}"

            if steps_per_second > 0:
                seconds = remaining_steps / steps_per_second
                if seconds < 3600:
                    eta_str = f"{seconds/60:.1f} min"
                elif seconds < 86400:
                    eta_str = f"{seconds/3600:.1f} hours"
                else:
                    eta_str = f"{seconds/86400:.1f} days"
            else:
                eta_str = "N/A"

        print(f"  loss={target:<5.1f} {step_str:>15} {remaining_str:>15} {eta_str:>20}")

    print("=" * 60)
    print(f"\n[NOTE] Power law model predicts loss converges to {c:.4f}.")
    print(f"       Targets below {c:.4f} are theoretically unreachable.")


def plot_loss(df: pd.DataFrame, output_path: str, smoothing_window: int = None):
    """Generate loss plot with log x-axis (step) and linear y-axis (loss)"""

    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Primary x-axis: global_step (log scale)
    steps = df['global_step'].values
    loss = df['loss'].values

    # Plot raw loss (all data)
    ax1.plot(steps, loss, alpha=0.6, color='blue', linewidth=1.0, label='Loss')

    # Smoothed loss (moving average) - only if specified
    if smoothing_window is not None and len(df) >= smoothing_window:
        smoothed = pd.Series(loss).rolling(window=smoothing_window, min_periods=1).mean()
        ax1.plot(steps, smoothed, color='red', linewidth=2,
                 label=f'Smoothed (MA-{smoothing_window})')

    # Fit and plot power law trend line
    a, b, c, r_squared = fit_power_law(steps, loss)
    if a is not None:
        # Plot fitted line
        fit_steps = np.logspace(np.log10(steps.min()), np.log10(steps.max() * 10), 100)
        fit_loss = a * np.power(fit_steps, -b) + c
        ax1.plot(fit_steps, fit_loss, '--', color='green', linewidth=2, alpha=0.7,
                 label=f'Fit: {a:.2f}*x^(-{b:.3f})+{c:.3f} (R²={r_squared:.3f})')
        # Plot asymptote
        ax1.axhline(y=c, color='orange', linestyle=':', linewidth=1.5, alpha=0.7,
                    label=f'Asymptote: {c:.4f}')

    # Log scale for x-axis, linear for y-axis
    ax1.set_xscale('log')

    # Y-axis: linear scale, fixed range for stable visualization
    min_y = loss.min()
    max_y = 1.1  # Fixed upper limit to clip spikes

    y_padding = (max_y - min_y) * 0.05
    ax1.set_ylim(min_y - y_padding, max_y + y_padding)

    ax1.set_xlabel('Global Step (log scale)', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(loc='upper right', fontsize=10)

    # Secondary x-axis: epoch markers
    ax2 = ax1.twiny()

    # Find epoch boundaries
    epoch_changes = df[df['epoch'] != df['epoch'].shift(1)]

    if len(epoch_changes) > 0:
        ax2.set_xscale('log')
        ax2.set_xlim(ax1.get_xlim())
        epoch_positions = epoch_changes['global_step'].values
        ax2.set_xticks(epoch_positions)
        ax2.set_xticklabels([f"E{int(e)}" for e in epoch_changes['epoch']], fontsize=10)
        ax2.set_xlabel('Epoch', fontsize=12)

    # Title with stats
    final_loss = loss[-1]
    min_loss = loss.min()
    max_step = steps.max()
    start_step = steps.min()

    title = f"Training Loss (Steps: {start_step:,} ~ {max_step:,}, Final: {final_loss:.4f}, Min: {min_loss:.4f})"
    ax1.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Print ETA predictions
    print_eta_predictions(df, max_step)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {output_path}")
    plt.close()


def plot_loss_detailed(df: pd.DataFrame, output_path: str):
    """Generate detailed loss analysis plot with 4 subplots"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    steps = df['global_step'].values
    loss = df['loss'].values

    # 1. Full loss curve (log scale) with epoch markers
    ax1 = axes[0, 0]
    ax1.plot(steps, loss, alpha=0.4, color='blue', linewidth=0.8)
    smoothed = pd.Series(loss).rolling(window=20, min_periods=1).mean()
    ax1.plot(steps, smoothed, color='red', linewidth=2, label='MA-20')

    # Add epoch boundary lines
    epoch_changes = df[df['epoch'] != df['epoch'].shift(1)]
    for _, row in epoch_changes.iterrows():
        ax1.axvline(x=row['global_step'], color='green', linestyle='--', alpha=0.5, linewidth=1)
        ax1.text(row['global_step'], ax1.get_ylim()[1] * 0.9, f"E{int(row['epoch'])}",
                fontsize=8, ha='left', color='green')

    ax1.set_xlabel('Global Step')
    ax1.set_ylabel('Loss (log)')
    ax1.set_title('Training Loss (Log Scale) with Epoch Markers')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3, which='both')

    # 2. Last 50% loss (linear scale for detail)
    ax2 = axes[0, 1]
    mid_idx = len(df) // 2
    df_last = df.iloc[mid_idx:]
    steps_last = df_last['global_step'].values
    loss_last = df_last['loss'].values

    ax2.plot(steps_last, loss_last, alpha=0.4, color='blue', linewidth=0.8)
    smoothed_last = pd.Series(loss_last).rolling(window=10, min_periods=1).mean()
    ax2.plot(steps_last, smoothed_last, color='red', linewidth=2, label='MA-10')
    ax2.set_xlabel('Global Step')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss (Last 50%, Linear Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Loss per epoch (box plot style - mean with std)
    ax3 = axes[1, 0]
    epoch_stats = df.groupby('epoch')['loss'].agg(['mean', 'std', 'min', 'max'])
    epochs = epoch_stats.index.values
    means = epoch_stats['mean'].values
    stds = epoch_stats['std'].values
    mins = epoch_stats['min'].values
    maxs = epoch_stats['max'].values

    ax3.errorbar(epochs, means, yerr=stds, fmt='o-', capsize=5, capthick=2,
                color='blue', label='Mean ± Std')
    ax3.fill_between(epochs, mins, maxs, alpha=0.2, color='blue', label='Min-Max Range')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.set_title('Loss Statistics per Epoch')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # 4. Learning rate schedule
    ax4 = axes[1, 1]
    ax4.plot(steps, df['lr'].values, color='green', linewidth=2)
    ax4.set_xlabel('Global Step')
    ax4.set_ylabel('Learning Rate')
    ax4.set_title('Learning Rate Schedule')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save with different name for detailed version
    detailed_path = output_path.replace('.png', '_detailed.png')
    plt.savefig(detailed_path, dpi=150, bbox_inches='tight')
    print(f"Saved detailed plot to {detailed_path}")
    plt.close()


def print_statistics(df: pd.DataFrame):
    """Print training statistics"""
    print("\n" + "=" * 60)
    print("Training Statistics (Latest Session)")
    print("=" * 60)

    print(f"\nTime range: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
    duration = df['timestamp'].max() - df['timestamp'].min()
    print(f"Duration: {duration}")

    print(f"\nTotal records: {len(df)}")
    print(f"Step range: {df['global_step'].min()} - {df['global_step'].max()}")
    print(f"Epochs: {df['epoch'].min()} - {df['epoch'].max()}")

    loss = df['loss'].values
    print(f"\nLoss statistics:")
    print(f"  Initial:  {loss[0]:.6f}")
    print(f"  Final:    {loss[-1]:.6f}")
    print(f"  Min:      {loss.min():.6f} (step {df.loc[df['loss'].idxmin(), 'global_step']})")
    print(f"  Max:      {loss.max():.6f} (step {df.loc[df['loss'].idxmax(), 'global_step']})")
    print(f"  Mean:     {loss.mean():.6f}")
    print(f"  Median:   {np.median(loss):.6f}")

    # Loss improvement (excluding outliers)
    # Use median of first/last 10 values for robustness
    initial_median = np.median(loss[:10])
    final_median = np.median(loss[-10:])
    improvement = (initial_median - final_median) / initial_median * 100
    print(f"\nLoss improvement: {improvement:.2f}% (median of first 10 vs last 10)")

    # Per-epoch stats
    print(f"\nPer-epoch summary:")
    epoch_stats = df.groupby('epoch')['loss'].agg(['mean', 'min', 'count'])
    for epoch, row in epoch_stats.iterrows():
        print(f"  Epoch {int(epoch)}: mean={row['mean']:.4f}, min={row['min']:.4f}, steps={int(row['count'])}")

    # Learning rate
    print(f"\nLearning rate: {df['lr'].min():.2e} - {df['lr'].max():.2e}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Visualize training loss from CSV')
    parser.add_argument('--input', '-i', type=str, default='./loss.csv',
                        help='Path to loss CSV file (default: ./loss.csv)')
    parser.add_argument('--output', '-o', type=str, default='loss.png',
                        help='Output image path (default: loss.png)')
    parser.add_argument('--smoothing', '-s', type=int, default=None,
                        help='Smoothing window size (default: None, no smoothing)')
    parser.add_argument('--detailed', '-d', action='store_true',
                        help='Generate detailed analysis plot')
    parser.add_argument('--all', '-a', action='store_true',
                        help='Use all sessions instead of just the latest')

    args = parser.parse_args()

    # Check input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")

        common_paths = [
            './loss.csv',
            './loss_log.csv',
            '/tmp/loss_log.csv',
            'outputs/loss.csv',
        ]

        for path in common_paths:
            if Path(path).exists():
                print(f"  Found: {path}")
                args.input = path
                break
        else:
            print("  No loss CSV found in common locations.")
            return

    # Load data
    df = load_loss_data(args.input)

    # Detect sessions
    df = detect_sessions(df)

    # Use latest session only (unless --all specified)
    if not args.all:
        df = get_latest_session(df)

    # Filter out extreme outliers (loss > 1000, likely initialization spikes)
    original_len = len(df)
    df_filtered = df[df['loss'] < 10000].copy()
    if len(df_filtered) < original_len:
        print(f"Filtered {original_len - len(df_filtered)} outlier records (loss > 10000)")
        df = df_filtered

    # Print statistics
    print_statistics(df)

    # Generate plots
    plot_loss(df, args.output, args.smoothing)

    if args.detailed:
        plot_loss_detailed(df, args.output)


if __name__ == "__main__":
    main()

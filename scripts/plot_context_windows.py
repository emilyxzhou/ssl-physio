"""
Plotting script for Context Windows MAML experiment results.

Generates publication-quality figures for academic presentation.
"""

import json
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
import seaborn as sns

# ============================================================================
# Configuration
# ============================================================================

# Results directory (handle nested structure)
RESULTS_BASE = Path("/data1/mjma/ssl-physio/context_windows_results/context_windows_results")
OUTPUT_DIR = Path("/data1/mjma/ssl-physio/context_windows_plots")

# Embedding configurations to plot
EMBEDDING_CONFIGS = {
    "s4_masking_30": "S4 (30%)",
    "s4_masking_70": "S4 (70%)",
    "mamba_masking_30": "Mamba (30%)",
    "mamba_masking_50": "Mamba (50%)",
    "raw_data": "Raw Signal",
}

# Regression targets
REGRESSION_TARGETS = ["rhr", "sleep", "steps"]
TARGET_DISPLAY = {
    "rhr": "Resting Heart Rate",
    "sleep": "Sleep Duration", 
    "steps": "Step Count"
}

# Plot parameters
INPUT_DAYS = [3, 5, 7]
OUTPUT_DAYS = [1, 2, 3, 4, 5, 6, 7]
SEEDS = [0, 1, 2, 3, 4]

# ============================================================================
# Color Palette - Academic/Publication Style
# ============================================================================

# Carefully curated palette for colorblind accessibility and print clarity
COLORS = {
    "raw_data": "#2C3E50",        # Dark slate (baseline)
    "s4_masking_30": "#E74C3C",   # Coral red
    "s4_masking_70": "#C0392B",   # Dark red  
    "mamba_masking_30": "#3498DB", # Sky blue
    "mamba_masking_50": "#2980B9", # Dark blue
}

MARKERS = {
    "raw_data": "s",              # Square (baseline)
    "s4_masking_30": "o",         # Circle
    "s4_masking_70": "^",         # Triangle up
    "mamba_masking_30": "D",      # Diamond
    "mamba_masking_50": "v",      # Triangle down
}

LINE_STYLES = {
    "raw_data": "-",
    "s4_masking_30": "-",
    "s4_masking_70": "--",
    "mamba_masking_30": "-",
    "mamba_masking_50": "--",
}

# ============================================================================
# Plot Style Configuration
# ============================================================================

def setup_plot_style():
    """Configure matplotlib for publication-quality plots."""
    plt.rcParams.update({
        # Font settings
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        
        # Figure settings
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        
        # Axes settings
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        
        # Line settings
        'lines.linewidth': 1.8,
        'lines.markersize': 7,
        
        # Legend settings
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
        'legend.fancybox': False,
    })

# ============================================================================
# Data Loading
# ============================================================================

def load_all_results():
    """
    Load all results from the results directory.
    
    Returns:
        dict: Nested dictionary [embedding_config][input_days][output_days][seed] = metrics
    """
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    for emb_config in EMBEDDING_CONFIGS.keys():
        emb_dir = RESULTS_BASE / emb_config
        if not emb_dir.exists():
            print(f"  Warning: {emb_config} not found")
            continue
            
        for in_days in INPUT_DAYS:
            for out_days in OUTPUT_DAYS:
                days_dir = emb_dir / f"in_{in_days:02d}_out_{out_days:02d}" / "cnn"
                
                for seed in SEEDS:
                    result_file = days_dir / f"seed_{seed}" / "results.json"
                    if result_file.exists():
                        with open(result_file, 'r') as f:
                            data = json.load(f)
                        results[emb_config][in_days][out_days][seed] = data.get("average_results", {})
    
    return results


def aggregate_results(results, metric_key):
    """
    Aggregate results across seeds for a specific metric.
    
    Returns:
        dict: [embedding_config][input_days][output_days] = (mean, std)
    """
    aggregated = defaultdict(lambda: defaultdict(dict))
    
    for emb_config in results:
        for in_days in results[emb_config]:
            for out_days in results[emb_config][in_days]:
                seed_values = []
                for seed, metrics in results[emb_config][in_days][out_days].items():
                    if metric_key in metrics:
                        seed_values.append(metrics[metric_key])
                
                if seed_values:
                    aggregated[emb_config][in_days][out_days] = (
                        np.mean(seed_values),
                        np.std(seed_values)
                    )
    
    return aggregated


def get_averaged_mse(results, avg_over="output_days"):
    """
    Get MSE averaged across targets and optionally across input/output days.
    
    Args:
        results: Raw results dict
        avg_over: "output_days" to average across forecast lengths,
                  "input_days" to average across context lengths,
                  "both" to get single value per embedding config
    
    Returns:
        dict: Aggregated MSE values
    """
    # First, compute combined MSE (average of rhr, sleep, steps)
    aggregated = defaultdict(lambda: defaultdict(dict))
    
    for emb_config in results:
        for in_days in results[emb_config]:
            for out_days in results[emb_config][in_days]:
                seed_mses = []
                for seed, metrics in results[emb_config][in_days][out_days].items():
                    target_mses = []
                    for target in REGRESSION_TARGETS:
                        key = f"{target}_mse_mean"
                        if key in metrics:
                            target_mses.append(metrics[key])
                    if len(target_mses) == 3:
                        seed_mses.append(np.mean(target_mses))
                
                if seed_mses:
                    aggregated[emb_config][in_days][out_days] = (
                        np.mean(seed_mses),
                        np.std(seed_mses) / np.sqrt(len(seed_mses))  # SEM
                    )
    
    return aggregated


# ============================================================================
# Plot 1: Context Window Length vs MSE
# ============================================================================

def plot_context_window_vs_mse(results, output_dir):
    """
    Plot context window length (input_days) vs MSE.
    Averages across all output_days and seeds.
    """
    setup_plot_style()
    aggregated = get_averaged_mse(results, avg_over="output_days")
    
    fig, ax = plt.subplots(figsize=(5.5, 4))
    
    for emb_config in ["raw_data", "s4_masking_30", "s4_masking_70", "mamba_masking_30", "mamba_masking_50"]:
        if emb_config not in aggregated:
            continue
            
        x_vals = []
        y_means = []
        y_stds = []
        
        for in_days in INPUT_DAYS:
            # Average across all output days
            out_means = []
            out_stds = []
            for out_days in OUTPUT_DAYS:
                if out_days in aggregated[emb_config][in_days]:
                    mean, std = aggregated[emb_config][in_days][out_days]
                    out_means.append(mean)
                    out_stds.append(std)
            
            if out_means:
                x_vals.append(in_days)
                y_means.append(np.mean(out_means))
                # Propagate uncertainty
                y_stds.append(np.sqrt(np.mean(np.array(out_stds)**2)))
        
        if x_vals:
            ax.errorbar(
                x_vals, y_means, yerr=y_stds,
                label=EMBEDDING_CONFIGS[emb_config],
                color=COLORS[emb_config],
                marker=MARKERS[emb_config],
                linestyle=LINE_STYLES[emb_config],
                capsize=3,
                capthick=1.2,
                markeredgecolor='white',
                markeredgewidth=0.8
            )
    
    ax.set_xlabel("Context Window Length (days)")
    ax.set_ylabel("Mean Squared Error")
    ax.set_xticks(INPUT_DAYS)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc='upper right', frameon=True)
    ax.set_title("Effect of Context Window Length on Prediction Error")
    
    plt.tight_layout()
    plt.savefig(output_dir / "pdf" / "context_window_vs_mse.pdf")
    plt.savefig(output_dir / "png" / "context_window_vs_mse.png")
    plt.close()
    print("  Saved: context_window_vs_mse")


# ============================================================================
# Plot 2: Forecast Window Length vs MSE (Two Panels)
# ============================================================================

def plot_forecast_window_vs_mse(results, output_dir):
    """
    Plot forecast window length (output_days) vs MSE.
    Two separate panels: S4 variants + baseline, Mamba variants + baseline.
    Both panels share the same y-axis range for consistency.
    """
    setup_plot_style()
    aggregated = get_averaged_mse(results)
    
    # Panel configurations
    panels = [
        {
            "title": "S4 Embeddings vs Raw Signal",
            "configs": ["raw_data", "s4_masking_30", "s4_masking_70"],
            "filename": "forecast_window_vs_mse_s4"
        },
        {
            "title": "Mamba Embeddings vs Raw Signal", 
            "configs": ["raw_data", "mamba_masking_30", "mamba_masking_50"],
            "filename": "forecast_window_vs_mse_mamba"
        }
    ]
    
    # Pre-compute all data to determine global y-axis limits
    all_panel_data = []
    global_y_min = float('inf')
    global_y_max = float('-inf')
    
    for panel in panels:
        panel_data = {}
        for emb_config in panel["configs"]:
            if emb_config not in aggregated:
                continue
            
            x_vals = []
            y_means = []
            y_stds = []
            
            for out_days in OUTPUT_DAYS:
                in_means = []
                in_stds = []
                for in_days in INPUT_DAYS:
                    if out_days in aggregated[emb_config][in_days]:
                        mean, std = aggregated[emb_config][in_days][out_days]
                        in_means.append(mean)
                        in_stds.append(std)
                
                if in_means:
                    x_vals.append(out_days)
                    y_means.append(np.mean(in_means))
                    y_stds.append(np.sqrt(np.mean(np.array(in_stds)**2)))
            
            if x_vals:
                panel_data[emb_config] = (x_vals, y_means, y_stds)
                # Update global limits (include error bars)
                for m, s in zip(y_means, y_stds):
                    global_y_min = min(global_y_min, m - s)
                    global_y_max = max(global_y_max, m + s)
        
        all_panel_data.append(panel_data)
    
    # Add padding to y-axis limits
    y_range = global_y_max - global_y_min
    y_padding = y_range * 0.1
    global_y_min = max(0, global_y_min - y_padding)  # Don't go below 0 for MSE
    global_y_max = global_y_max + y_padding
    
    # Now create plots with consistent axes
    for panel, panel_data in zip(panels, all_panel_data):
        fig, ax = plt.subplots(figsize=(5.5, 4))
        
        for emb_config in panel["configs"]:
            if emb_config not in panel_data:
                continue
            
            x_vals, y_means, y_stds = panel_data[emb_config]
            ax.errorbar(
                x_vals, y_means, yerr=y_stds,
                label=EMBEDDING_CONFIGS[emb_config],
                color=COLORS[emb_config],
                marker=MARKERS[emb_config],
                linestyle=LINE_STYLES[emb_config],
                capsize=3,
                capthick=1.2,
                markeredgecolor='white',
                markeredgewidth=0.8
            )
        
        ax.set_xlabel("Forecast Horizon (days)")
        ax.set_ylabel("Mean Squared Error")
        ax.set_xticks(OUTPUT_DAYS)
        ax.set_ylim(global_y_min, global_y_max)  # Apply consistent y-axis
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(loc='upper left', frameon=True)
        ax.set_title(panel["title"])
        
        plt.tight_layout()
        plt.savefig(output_dir / "pdf" / f"{panel['filename']}.pdf")
        plt.savefig(output_dir / "png" / f"{panel['filename']}.png")
        plt.close()
        print(f"  Saved: {panel['filename']}")


# ============================================================================
# Plot 3: Heatmaps
# ============================================================================

def plot_heatmaps(results, output_dir):
    """
    Plot heatmaps of MSE for context window × forecast window.
    One heatmap per embedding configuration.
    """
    setup_plot_style()
    aggregated = get_averaged_mse(results)
    
    # Compute global min/max for consistent colorbar
    all_values = []
    for emb_config in EMBEDDING_CONFIGS.keys():
        if emb_config not in aggregated:
            continue
        for in_days in INPUT_DAYS:
            for out_days in OUTPUT_DAYS:
                if out_days in aggregated[emb_config][in_days]:
                    all_values.append(aggregated[emb_config][in_days][out_days][0])
    
    if not all_values:
        print("  Warning: No data for heatmaps")
        return
    
    vmin, vmax = min(all_values), max(all_values)
    
    # Create individual heatmaps
    for emb_config in EMBEDDING_CONFIGS.keys():
        if emb_config not in aggregated:
            continue
        
        # Build matrix
        matrix = np.full((len(INPUT_DAYS), len(OUTPUT_DAYS)), np.nan)
        
        for i, in_days in enumerate(INPUT_DAYS):
            for j, out_days in enumerate(OUTPUT_DAYS):
                if out_days in aggregated[emb_config][in_days]:
                    matrix[i, j] = aggregated[emb_config][in_days][out_days][0]
        
        fig, ax = plt.subplots(figsize=(6, 3.5))
        
        # Use a sequential colormap that works in grayscale
        cmap = sns.color_palette("rocket_r", as_cmap=True)
        
        im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('MSE', rotation=270, labelpad=15)
        
        # Set ticks
        ax.set_xticks(np.arange(len(OUTPUT_DAYS)))
        ax.set_yticks(np.arange(len(INPUT_DAYS)))
        ax.set_xticklabels(OUTPUT_DAYS)
        ax.set_yticklabels(INPUT_DAYS)
        
        # Labels
        ax.set_xlabel("Forecast Horizon (days)")
        ax.set_ylabel("Context Window (days)")
        ax.set_title(f"{EMBEDDING_CONFIGS[emb_config]}")
        
        # Add value annotations
        for i in range(len(INPUT_DAYS)):
            for j in range(len(OUTPUT_DAYS)):
                if not np.isnan(matrix[i, j]):
                    # Choose text color based on background
                    text_color = 'white' if matrix[i, j] > (vmin + vmax) / 2 else 'black'
                    ax.text(j, i, f'{matrix[i, j]:.3f}',
                           ha='center', va='center', color=text_color,
                           fontsize=8, fontweight='medium')
        
        plt.tight_layout()
        plt.savefig(output_dir / "pdf" / f"heatmap_{emb_config}.pdf")
        plt.savefig(output_dir / "png" / f"heatmap_{emb_config}.png")
        plt.close()
        print(f"  Saved: heatmap_{emb_config}")
    
    # Create combined heatmap figure (2x3 grid)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    axes = axes.flatten()
    
    configs_to_plot = ["raw_data", "s4_masking_30", "s4_masking_70", 
                       "mamba_masking_30", "mamba_masking_50"]
    
    cmap = sns.color_palette("rocket_r", as_cmap=True)
    
    for idx, emb_config in enumerate(configs_to_plot):
        ax = axes[idx]
        
        if emb_config not in aggregated:
            ax.set_visible(False)
            continue
        
        matrix = np.full((len(INPUT_DAYS), len(OUTPUT_DAYS)), np.nan)
        for i, in_days in enumerate(INPUT_DAYS):
            for j, out_days in enumerate(OUTPUT_DAYS):
                if out_days in aggregated[emb_config][in_days]:
                    matrix[i, j] = aggregated[emb_config][in_days][out_days][0]
        
        im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        
        ax.set_xticks(np.arange(len(OUTPUT_DAYS)))
        ax.set_yticks(np.arange(len(INPUT_DAYS)))
        ax.set_xticklabels(OUTPUT_DAYS)
        ax.set_yticklabels(INPUT_DAYS)
        
        if idx >= 3:  # Bottom row
            ax.set_xlabel("Forecast Horizon (days)")
        if idx % 3 == 0:  # Left column
            ax.set_ylabel("Context Window (days)")
        
        ax.set_title(EMBEDDING_CONFIGS[emb_config], fontsize=10)
        
        # Add value annotations (smaller font for combined)
        for i in range(len(INPUT_DAYS)):
            for j in range(len(OUTPUT_DAYS)):
                if not np.isnan(matrix[i, j]):
                    text_color = 'white' if matrix[i, j] > (vmin + vmax) / 2 else 'black'
                    ax.text(j, i, f'{matrix[i, j]:.2f}',
                           ha='center', va='center', color=text_color,
                           fontsize=7)
    
    # Hide unused subplot
    axes[-1].set_visible(False)
    
    # Add shared colorbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Mean Squared Error', rotation=270, labelpad=15)
    
    plt.suptitle("Prediction Error Across Window Configurations", y=0.98, fontsize=12)
    plt.savefig(output_dir / "pdf" / "heatmaps_combined.pdf", bbox_inches='tight')
    plt.savefig(output_dir / "png" / "heatmaps_combined.png", bbox_inches='tight')
    plt.close()
    print("  Saved: heatmaps_combined")


# ============================================================================
# Plot 4: Per-Target Comparison
# ============================================================================

def plot_per_target_comparison(results, output_dir):
    """
    Plot MSE comparison across targets (RHR, Sleep, Steps) for each embedding config.
    Bar chart format for clear comparison.
    """
    setup_plot_style()
    
    # Aggregate by target
    target_data = defaultdict(lambda: defaultdict(list))
    
    for emb_config in results:
        for in_days in results[emb_config]:
            for out_days in results[emb_config][in_days]:
                for seed, metrics in results[emb_config][in_days][out_days].items():
                    for target in REGRESSION_TARGETS:
                        key = f"{target}_mse_mean"
                        if key in metrics:
                            target_data[emb_config][target].append(metrics[key])
    
    # Compute mean and std per embedding config and target
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    x = np.arange(len(REGRESSION_TARGETS))
    width = 0.15
    offsets = np.linspace(-2*width, 2*width, 5)
    
    configs_to_plot = ["raw_data", "s4_masking_30", "s4_masking_70", 
                       "mamba_masking_30", "mamba_masking_50"]
    
    for i, emb_config in enumerate(configs_to_plot):
        if emb_config not in target_data:
            continue
        
        means = []
        stds = []
        for target in REGRESSION_TARGETS:
            values = target_data[emb_config][target]
            if values:
                means.append(np.mean(values))
                stds.append(np.std(values) / np.sqrt(len(values)))
            else:
                means.append(0)
                stds.append(0)
        
        bars = ax.bar(x + offsets[i], means, width, 
                     label=EMBEDDING_CONFIGS[emb_config],
                     color=COLORS[emb_config],
                     edgecolor='white',
                     linewidth=0.5)
        ax.errorbar(x + offsets[i], means, yerr=stds,
                   fmt='none', color='black', capsize=2, capthick=1, linewidth=1)
    
    ax.set_xlabel("Prediction Target")
    ax.set_ylabel("Mean Squared Error")
    ax.set_xticks(x)
    ax.set_xticklabels([TARGET_DISPLAY[t] for t in REGRESSION_TARGETS])
    ax.legend(loc='upper right', frameon=True, ncol=2)
    ax.set_title("Prediction Error by Target Variable")
    
    plt.tight_layout()
    plt.savefig(output_dir / "pdf" / "per_target_comparison.pdf")
    plt.savefig(output_dir / "png" / "per_target_comparison.png")
    plt.close()
    print("  Saved: per_target_comparison")


# ============================================================================
# Main
# ============================================================================

def main():
    """Generate all plots."""
    print("\n" + "="*60)
    print("Context Windows Results Plotting")
    print("="*60)
    
    # Create output directories (pdf and png subfolders)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "pdf").mkdir(exist_ok=True)
    (OUTPUT_DIR / "png").mkdir(exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"  PDFs: {OUTPUT_DIR / 'pdf'}")
    print(f"  PNGs: {OUTPUT_DIR / 'png'}")
    
    # Load results
    print("\nLoading results...")
    results = load_all_results()
    
    # Count loaded results
    total = sum(
        len(results[e][i][o]) 
        for e in results for i in results[e] for o in results[e][i]
    )
    print(f"  Loaded {total} result files")
    
    # Generate plots
    print("\nGenerating plots...")
    
    print("\n[1/4] Context window vs MSE...")
    plot_context_window_vs_mse(results, OUTPUT_DIR)
    
    print("\n[2/4] Forecast window vs MSE...")
    plot_forecast_window_vs_mse(results, OUTPUT_DIR)
    
    print("\n[3/4] Heatmaps...")
    plot_heatmaps(results, OUTPUT_DIR)
    
    print("\n[4/4] Per-target comparison...")
    plot_per_target_comparison(results, OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("All plots generated successfully!")
    print(f"Output: {OUTPUT_DIR}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()


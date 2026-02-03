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
    "mamba_masking_70": "Mamba (70%)",
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
    "mamba_masking_70": "#2980B9", # Dark blue
}

MARKERS = {
    "raw_data": "s",              # Square (baseline)
    "s4_masking_30": "o",         # Circle
    "s4_masking_70": "^",         # Triangle up
    "mamba_masking_30": "D",      # Diamond
    "mamba_masking_70": "v",      # Triangle down
}

LINE_STYLES = {
    "raw_data": "-",
    "s4_masking_30": "-",
    "s4_masking_70": "--",
    "mamba_masking_30": "-",
    "mamba_masking_70": "--",
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


def get_averaged_mse(results):
    """
    Get MSE aggregated per (emb_config, input_days, output_days) combination.
    
    For each combination:
    - Average MSE across the 3 regression targets (rhr, sleep, steps)
    - Compute mean and SEM across seeds
    
    Returns:
        dict: aggregated[emb_config][input_days][output_days] = (mean, sem)
    """
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
    aggregated = get_averaged_mse(results)
    
    fig, ax = plt.subplots(figsize=(5.5, 4))
    
    for emb_config in ["raw_data", "s4_masking_30", "s4_masking_70", "mamba_masking_30", "mamba_masking_70"]:
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
    Single combined plot with all embedding configurations.
    """
    setup_plot_style()
    aggregated = get_averaged_mse(results)
    
    # All configs to plot
    all_configs = ["raw_data", "s4_masking_30", "s4_masking_70", "mamba_masking_30", "mamba_masking_70"]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for emb_config in all_configs:
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
    ax.set_title("Forecast Window vs MSE")
    ax.set_xticks(OUTPUT_DAYS)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc='best', frameon=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "pdf" / "forecast_window_vs_mse.pdf")
    plt.savefig(output_dir / "png" / "forecast_window_vs_mse.png")
    plt.close()
    print("  Saved: forecast_window_vs_mse")


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
                       "mamba_masking_30", "mamba_masking_70"]
    
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
                       "mamba_masking_30", "mamba_masking_70"]
    
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
# Plot 5: Seed Variability Analysis
# ============================================================================

def plot_seed_variability(results, output_dir):
    """
    Plot seed-to-seed variability to show how random subject selection affects results.
    Creates multiple visualizations:
    1. Box plot of MSE across seeds for each configuration
    2. Coefficient of variation by configuration
    3. Seed-wise scatter showing individual seed results
    """
    setup_plot_style()
    
    # Collect all seed MSEs per configuration
    seed_data = defaultdict(lambda: defaultdict(list))  # config -> seed -> [mses]
    config_all_mses = defaultdict(list)  # config -> [all mses across all conditions]
    
    for emb_config in results:
        for in_days in results[emb_config]:
            for out_days in results[emb_config][in_days]:
                for seed, metrics in results[emb_config][in_days][out_days].items():
                    target_mses = []
                    for target in REGRESSION_TARGETS:
                        key = f"{target}_mse_mean"
                        if key in metrics:
                            target_mses.append(metrics[key])
                    if len(target_mses) == 3:
                        avg_mse = np.mean(target_mses)
                        seed_data[emb_config][seed].append(avg_mse)
                        config_all_mses[emb_config].append(avg_mse)
    
    configs_to_plot = ["raw_data", "s4_masking_30", "s4_masking_70", 
                       "mamba_masking_30", "mamba_masking_70"]
    configs_present = [c for c in configs_to_plot if c in seed_data]
    
    # -------------------------------------------------------------------------
    # Plot 5a: Box plot showing MSE distribution across seeds
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    box_data = []
    box_labels = []
    box_colors = []
    
    for emb_config in configs_present:
        # Compute per-seed mean MSE (each seed has multiple conditions)
        seed_means = []
        for seed in SEEDS:
            if seed in seed_data[emb_config] and seed_data[emb_config][seed]:
                seed_means.append(np.mean(seed_data[emb_config][seed]))
        if seed_means:
            box_data.append(seed_means)
            box_labels.append(EMBEDDING_CONFIGS[emb_config])
            box_colors.append(COLORS[emb_config])
    
    if box_data:
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, 
                       widths=0.6, showfliers=True, notch=False)
        
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for whisker in bp['whiskers']:
            whisker.set_color('#555555')
        for cap in bp['caps']:
            cap.set_color('#555555')
        for median in bp['medians']:
            median.set_color('#222222')
            median.set_linewidth(2)
        
        # Overlay individual seed points
        for i, (data, color) in enumerate(zip(box_data, box_colors)):
            x_jitter = np.random.normal(i+1, 0.04, len(data))
            ax.scatter(x_jitter, data, color=color, s=50, alpha=0.8, 
                      edgecolor='white', linewidth=1, zorder=10)
    
    ax.set_ylabel("Mean MSE (averaged over conditions)")
    ax.set_title("Seed Variability: MSE Distribution Across Random Seeds")
    plt.xticks(rotation=15, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_dir / "pdf" / "seed_variability_boxplot.pdf")
    plt.savefig(output_dir / "png" / "seed_variability_boxplot.png")
    plt.close()
    print("  Saved: seed_variability_boxplot")
    
    # -------------------------------------------------------------------------
    # Plot 5b: Coefficient of Variation (CV) by configuration
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    
    cvs = []
    cv_labels = []
    cv_colors = []
    
    for emb_config in configs_present:
        seed_means = []
        for seed in SEEDS:
            if seed in seed_data[emb_config] and seed_data[emb_config][seed]:
                seed_means.append(np.mean(seed_data[emb_config][seed]))
        if len(seed_means) >= 2:
            cv = np.std(seed_means) / np.mean(seed_means) * 100  # CV as percentage
            cvs.append(cv)
            cv_labels.append(EMBEDDING_CONFIGS[emb_config])
            cv_colors.append(COLORS[emb_config])
    
    if cvs:
        x_pos = np.arange(len(cvs))
        bars = ax.bar(x_pos, cvs, color=cv_colors, edgecolor='white', linewidth=1)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(cv_labels, rotation=15, ha='right')
        ax.set_ylabel("Coefficient of Variation (%)")
        ax.set_title("Seed Variability: How Much Does Random Selection Matter?")
        
        # Add value labels on bars
        for bar, cv in zip(bars, cvs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                   f'{cv:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "pdf" / "seed_variability_cv.pdf")
    plt.savefig(output_dir / "png" / "seed_variability_cv.png")
    plt.close()
    print("  Saved: seed_variability_cv")
    
    # -------------------------------------------------------------------------
    # Plot 5c: Seed-wise line plot showing consistency across seeds
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    for emb_config in configs_present:
        seed_means = []
        seed_ids = []
        for seed in SEEDS:
            if seed in seed_data[emb_config] and seed_data[emb_config][seed]:
                seed_means.append(np.mean(seed_data[emb_config][seed]))
                seed_ids.append(seed)
        
        if seed_means:
            ax.plot(seed_ids, seed_means, 
                   label=EMBEDDING_CONFIGS[emb_config],
                   color=COLORS[emb_config],
                   marker=MARKERS[emb_config],
                   linestyle=LINE_STYLES[emb_config],
                   markeredgecolor='white',
                   markeredgewidth=0.8,
                   linewidth=1.5,
                   markersize=8)
    
    ax.set_xlabel("Seed")
    ax.set_ylabel("Mean MSE")
    ax.set_xticks(SEEDS)
    ax.set_title("Seed-wise Comparison: Individual Random Seed Performance")
    ax.legend(loc='upper right', frameon=True)
    
    plt.tight_layout()
    plt.savefig(output_dir / "pdf" / "seed_variability_lineplot.pdf")
    plt.savefig(output_dir / "png" / "seed_variability_lineplot.png")
    plt.close()
    print("  Saved: seed_variability_lineplot")
    
    # -------------------------------------------------------------------------
    # Plot 5d: Variability by input/output days
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Left: Variability by input days
    ax = axes[0]
    for emb_config in configs_present:
        in_cvs = []
        in_days_list = []
        for in_days in INPUT_DAYS:
            seed_vals = defaultdict(list)
            for out_days in OUTPUT_DAYS:
                if out_days in results[emb_config].get(in_days, {}):
                    for seed, metrics in results[emb_config][in_days][out_days].items():
                        target_mses = []
                        for target in REGRESSION_TARGETS:
                            key = f"{target}_mse_mean"
                            if key in metrics:
                                target_mses.append(metrics[key])
                        if len(target_mses) == 3:
                            seed_vals[seed].append(np.mean(target_mses))
            
            # Compute CV across seeds for this input_days
            seed_means = [np.mean(v) for v in seed_vals.values() if v]
            if len(seed_means) >= 2:
                cv = np.std(seed_means) / np.mean(seed_means) * 100
                in_cvs.append(cv)
                in_days_list.append(in_days)
        
        if in_cvs:
            ax.plot(in_days_list, in_cvs,
                   label=EMBEDDING_CONFIGS[emb_config],
                   color=COLORS[emb_config],
                   marker=MARKERS[emb_config],
                   linestyle=LINE_STYLES[emb_config],
                   markeredgecolor='white',
                   markeredgewidth=0.8)
    
    ax.set_xlabel("Context Window (days)")
    ax.set_ylabel("Coefficient of Variation (%)")
    ax.set_title("Variability by Context Length")
    ax.set_xticks(INPUT_DAYS)
    ax.legend(loc='best', frameon=True, fontsize=8)
    
    # Right: Variability by output days
    ax = axes[1]
    for emb_config in configs_present:
        out_cvs = []
        out_days_list = []
        for out_days in OUTPUT_DAYS:
            seed_vals = defaultdict(list)
            for in_days in INPUT_DAYS:
                if out_days in results[emb_config].get(in_days, {}):
                    for seed, metrics in results[emb_config][in_days][out_days].items():
                        target_mses = []
                        for target in REGRESSION_TARGETS:
                            key = f"{target}_mse_mean"
                            if key in metrics:
                                target_mses.append(metrics[key])
                        if len(target_mses) == 3:
                            seed_vals[seed].append(np.mean(target_mses))
            
            # Compute CV across seeds for this output_days
            seed_means = [np.mean(v) for v in seed_vals.values() if v]
            if len(seed_means) >= 2:
                cv = np.std(seed_means) / np.mean(seed_means) * 100
                out_cvs.append(cv)
                out_days_list.append(out_days)
        
        if out_cvs:
            ax.plot(out_days_list, out_cvs,
                   label=EMBEDDING_CONFIGS[emb_config],
                   color=COLORS[emb_config],
                   marker=MARKERS[emb_config],
                   linestyle=LINE_STYLES[emb_config],
                   markeredgecolor='white',
                   markeredgewidth=0.8)
    
    ax.set_xlabel("Forecast Horizon (days)")
    ax.set_ylabel("Coefficient of Variation (%)")
    ax.set_title("Variability by Forecast Length")
    ax.set_xticks(OUTPUT_DAYS)
    ax.legend(loc='best', frameon=True, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / "pdf" / "seed_variability_by_days.pdf")
    plt.savefig(output_dir / "png" / "seed_variability_by_days.png")
    plt.close()
    print("  Saved: seed_variability_by_days")
    
    # -------------------------------------------------------------------------
    # Plot 5e: Combined seed distribution by forecast day (line plot with error bars)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for emb_config in configs_present:
        means = []
        stds = []
        out_days_list = []
        
        for out_days in OUTPUT_DAYS:
            seed_means = []
            for seed in SEEDS:
                seed_mses = []
                for in_days in INPUT_DAYS:
                    if out_days in results[emb_config].get(in_days, {}):
                        if seed in results[emb_config][in_days][out_days]:
                            metrics = results[emb_config][in_days][out_days][seed]
                            target_mses = []
                            for target in REGRESSION_TARGETS:
                                key = f"{target}_mse_mean"
                                if key in metrics:
                                    target_mses.append(metrics[key])
                            if len(target_mses) == 3:
                                seed_mses.append(np.mean(target_mses))
                if seed_mses:
                    seed_means.append(np.mean(seed_mses))
            
            if len(seed_means) >= 2:
                means.append(np.mean(seed_means))
                stds.append(np.std(seed_means))
                out_days_list.append(out_days)
        
        if means:
            ax.errorbar(out_days_list, means, yerr=stds,
                       label=EMBEDDING_CONFIGS[emb_config],
                       color=COLORS[emb_config],
                       marker=MARKERS[emb_config],
                       linestyle=LINE_STYLES[emb_config],
                       markeredgecolor='white',
                       markeredgewidth=0.8,
                       capsize=3,
                       capthick=1.5,
                       elinewidth=1.5)
    
    ax.set_xlabel("Forecast Horizon (days)")
    ax.set_ylabel("MSE (mean ± std across seeds)")
    ax.set_title("Seed Variability by Forecast Day")
    ax.set_xticks(OUTPUT_DAYS)
    ax.legend(loc='best', frameon=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "pdf" / "seed_distribution_by_forecast.pdf")
    plt.savefig(output_dir / "png" / "seed_distribution_by_forecast.png")
    plt.close()
    print("  Saved: seed_distribution_by_forecast")


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
    
    print("\n[1/5] Context window vs MSE...")
    plot_context_window_vs_mse(results, OUTPUT_DIR)
    
    print("\n[2/5] Forecast window vs MSE...")
    plot_forecast_window_vs_mse(results, OUTPUT_DIR)
    
    print("\n[3/5] Heatmaps...")
    plot_heatmaps(results, OUTPUT_DIR)
    
    print("\n[4/5] Per-target comparison...")
    plot_per_target_comparison(results, OUTPUT_DIR)
    
    print("\n[5/5] Seed variability analysis...")
    plot_seed_variability(results, OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("All plots generated successfully!")
    print(f"Output: {OUTPUT_DIR}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()


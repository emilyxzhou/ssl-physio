#!/usr/bin/env python3
"""
Plotting script for embedding sequencers experiment results.
Generates visualizations comparing performance across different experimental conditions.

Parameters explored:
- Masking ratio: 10%, 30%, 50%
- Days given (context): 3, 5, 7
- Days predicted (horizon): 1, 5, 7, 14
- Model type: NN, CNN
"""

import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Set style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14

# Color palettes
MASKING_COLORS = {'10': '#3498db', '30': '#e74c3c', '50': '#2ecc71'}
MODEL_COLORS = {'nn': '#9b59b6', 'cnn': '#f39c12'}
METRIC_COLORS = {'bpm': '#e74c3c', 'steps': '#3498db', 'both': '#2ecc71'}


def load_all_results(base_dir: str) -> pd.DataFrame:
    """Load all results.json files into a pandas DataFrame."""
    records = []
    base_path = Path(base_dir)
    
    masking_dirs = ['masking_10', 'masking_30', 'masking_50']
    
    for masking_dir in masking_dirs:
        masking_path = base_path / masking_dir
        if not masking_path.exists():
            continue
            
        for given_dir in masking_path.iterdir():
            if not given_dir.is_dir() or not given_dir.name.startswith('given_'):
                continue
                
            for model_dir in given_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                    
                results_file = model_dir / 'results.json'
                if not results_file.exists():
                    continue
                    
                with open(results_file, 'r') as f:
                    data = json.load(f)
                
                params = data['parameters']
                avg_results = data['average_results']
                
                record = {
                    'masking_ratio': int(params['masking_model'].replace('masking_', '')),
                    'days_given': params['days_given'],
                    'days_predicted': params['days_predicted'],
                    'model_type': params['prediction_model'],
                    'train_bpm_loss': avg_results['train_bpm_loss'],
                    'train_steps_loss': avg_results['train_steps_loss'],
                    'train_both_loss': avg_results['train_both_loss'],
                    'test_bpm_loss': avg_results['test_bpm_loss'],
                    'test_steps_loss': avg_results['test_steps_loss'],
                    'test_both_loss': avg_results['test_both_loss'],
                    'num_subjects': data['metadata']['num_subjects'],
                }
                records.append(record)
    
    df = pd.DataFrame(records)
    return df.sort_values(['masking_ratio', 'days_given', 'days_predicted', 'model_type']).reset_index(drop=True)


def plot_model_comparison(df: pd.DataFrame, output_dir: str):
    """Plot NN vs CNN comparison across all conditions."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    metrics = [
        ('test_bpm_loss', 'BPM Loss (MSE)', 'Heart Rate Prediction'),
        ('test_steps_loss', 'Steps Loss (MAE)', 'Step Count Prediction'),
        ('test_both_loss', 'Combined Loss', 'Combined Prediction')
    ]
    
    for ax, (metric, ylabel, title) in zip(axes, metrics):
        nn_data = df[df['model_type'] == 'nn'].groupby(['masking_ratio', 'days_given'])[metric].mean().reset_index()
        cnn_data = df[df['model_type'] == 'cnn'].groupby(['masking_ratio', 'days_given'])[metric].mean().reset_index()
        
        x = np.arange(len(nn_data))
        width = 0.35
        
        labels = [f"M{row['masking_ratio']}% G{row['days_given']}d" 
                  for _, row in nn_data.iterrows()]
        
        bars1 = ax.bar(x - width/2, nn_data[metric], width, label='NN', 
                       color=MODEL_COLORS['nn'], alpha=0.85, edgecolor='white')
        bars2 = ax.bar(x + width/2, cnn_data[metric], width, label='CNN', 
                       color=MODEL_COLORS['cnn'], alpha=0.85, edgecolor='white')
        
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
    
    fig.suptitle('Model Architecture Comparison: NN vs CNN', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_model_comparison.png'), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: 01_model_comparison.png")


def plot_masking_effect(df: pd.DataFrame, output_dir: str):
    """Plot effect of masking ratio on test losses."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    metrics = [
        ('test_bpm_loss', 'BPM Loss (MSE)'),
        ('test_steps_loss', 'Steps Loss (MAE)'),
        ('test_both_loss', 'Combined Loss')
    ]
    
    masking_ratios = sorted(df['masking_ratio'].unique())
    
    for ax, (metric, ylabel) in zip(axes, metrics):
        # Group by masking ratio and model type
        for model in ['nn', 'cnn']:
            model_df = df[df['model_type'] == model]
            means = model_df.groupby('masking_ratio')[metric].mean()
            stds = model_df.groupby('masking_ratio')[metric].std()
            
            ax.errorbar(masking_ratios, means, yerr=stds, 
                       marker='o', markersize=8, linewidth=2, capsize=5,
                       label=model.upper(), color=MODEL_COLORS[model])
        
        ax.set_xlabel('Masking Ratio (%)')
        ax.set_ylabel(ylabel)
        ax.set_xticks(masking_ratios)
        ax.legend()
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
    
    fig.suptitle('Effect of Pre-training Masking Ratio on Test Loss', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_masking_effect.png'), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: 02_masking_effect.png")


def plot_prediction_horizon_effect(df: pd.DataFrame, output_dir: str):
    """Plot effect of prediction horizon (days_predicted) on test losses."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    metrics = [
        ('test_bpm_loss', 'BPM Loss (MSE)'),
        ('test_steps_loss', 'Steps Loss (MAE)'),
        ('test_both_loss', 'Combined Loss')
    ]
    
    days_predicted = sorted(df['days_predicted'].unique())
    
    for ax, (metric, ylabel) in zip(axes, metrics):
        for masking in sorted(df['masking_ratio'].unique()):
            mask_df = df[df['masking_ratio'] == masking]
            means = mask_df.groupby('days_predicted')[metric].mean()
            stds = mask_df.groupby('days_predicted')[metric].std()
            
            ax.errorbar(days_predicted, means, yerr=stds,
                       marker='s', markersize=7, linewidth=2, capsize=4,
                       label=f'Mask {masking}%', color=MASKING_COLORS[str(masking)])
        
        ax.set_xlabel('Prediction Horizon (days)')
        ax.set_ylabel(ylabel)
        ax.set_xticks(days_predicted)
        ax.legend()
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
    
    fig.suptitle('Effect of Prediction Horizon on Test Loss (by Masking Ratio)', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_prediction_horizon.png'), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: 03_prediction_horizon.png")


def plot_context_length_effect(df: pd.DataFrame, output_dir: str):
    """Plot effect of context length (days_given) on test losses."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    metrics = [
        ('test_bpm_loss', 'BPM Loss (MSE)'),
        ('test_steps_loss', 'Steps Loss (MAE)'),
        ('test_both_loss', 'Combined Loss')
    ]
    
    days_given = sorted(df['days_given'].unique())
    
    for ax, (metric, ylabel) in zip(axes, metrics):
        for masking in sorted(df['masking_ratio'].unique()):
            mask_df = df[df['masking_ratio'] == masking]
            means = mask_df.groupby('days_given')[metric].mean()
            stds = mask_df.groupby('days_given')[metric].std()
            
            ax.errorbar(days_given, means, yerr=stds,
                       marker='^', markersize=8, linewidth=2, capsize=4,
                       label=f'Mask {masking}%', color=MASKING_COLORS[str(masking)])
        
        ax.set_xlabel('Context Length (days)')
        ax.set_ylabel(ylabel)
        ax.set_xticks(days_given)
        ax.legend()
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
    
    fig.suptitle('Effect of Context Window Length on Test Loss (by Masking Ratio)', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_context_length.png'), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: 04_context_length.png")


def plot_heatmaps(df: pd.DataFrame, output_dir: str):
    """Plot heatmaps of test_both_loss for days_given x days_predicted."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    
    masking_ratios = sorted(df['masking_ratio'].unique())
    model_types = ['nn', 'cnn']
    
    # Custom colormap
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    
    # Find global min/max for consistent colorbar
    vmin = df['test_both_loss'].min()
    vmax = df['test_both_loss'].max()
    
    for row, model in enumerate(model_types):
        for col, masking in enumerate(masking_ratios):
            ax = axes[row, col]
            
            subset = df[(df['masking_ratio'] == masking) & (df['model_type'] == model)]
            pivot = subset.pivot(index='days_given', columns='days_predicted', values='test_both_loss')
            
            # Sort index and columns
            pivot = pivot.sort_index()
            pivot = pivot[sorted(pivot.columns)]
            
            sns.heatmap(pivot, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
                       annot=True, fmt='.0f', cbar=col == 2,
                       cbar_kws={'label': 'Combined Loss'} if col == 2 else {},
                       linewidths=0.5, linecolor='white')
            
            ax.set_title(f'{model.upper()} - Mask {masking}%')
            ax.set_xlabel('Days Predicted')
            ax.set_ylabel('Days Given')
    
    fig.suptitle('Test Combined Loss: Context × Horizon × Masking × Model', fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_heatmaps.png'), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: 05_heatmaps.png")


def plot_train_test_comparison(df: pd.DataFrame, output_dir: str):
    """Plot train vs test loss to check for overfitting."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    metrics = [
        ('train_bpm_loss', 'test_bpm_loss', 'BPM Loss'),
        ('train_steps_loss', 'test_steps_loss', 'Steps Loss'),
        ('train_both_loss', 'test_both_loss', 'Combined Loss')
    ]
    
    for ax, (train_col, test_col, title) in zip(axes, metrics):
        for model in ['nn', 'cnn']:
            model_df = df[df['model_type'] == model]
            ax.scatter(model_df[train_col], model_df[test_col], 
                      alpha=0.7, s=60, label=model.upper(),
                      color=MODEL_COLORS[model], edgecolors='white', linewidth=0.5)
        
        # Add diagonal line (perfect generalization)
        all_vals = np.concatenate([df[train_col].values, df[test_col].values])
        lims = [min(all_vals) * 0.9, max(all_vals) * 1.1]
        ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1.5, label='y=x')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        
        ax.set_xlabel(f'Train {title}')
        ax.set_ylabel(f'Test {title}')
        ax.set_title(title)
        ax.legend(loc='upper left')
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
    
    fig.suptitle('Train vs Test Loss (Generalization Check)', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '06_train_test_comparison.png'), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: 06_train_test_comparison.png")


def plot_comprehensive_bar(df: pd.DataFrame, output_dir: str):
    """Plot comprehensive bar chart of all configurations for combined loss."""
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Sort by test_both_loss for easy comparison
    df_sorted = df.sort_values('test_both_loss')
    
    # Create labels
    labels = [f"M{row['masking_ratio']}% G{row['days_given']}d P{row['days_predicted']}d {row['model_type'].upper()}"
              for _, row in df_sorted.iterrows()]
    
    x = np.arange(len(labels))
    colors = [MODEL_COLORS[row['model_type']] for _, row in df_sorted.iterrows()]
    
    bars = ax.bar(x, df_sorted['test_both_loss'], color=colors, alpha=0.85, edgecolor='white')
    
    ax.set_ylabel('Test Combined Loss')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, ha='center', fontsize=7)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add legend manually
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=MODEL_COLORS['nn'], label='NN', alpha=0.85),
                      Patch(facecolor=MODEL_COLORS['cnn'], label='CNN', alpha=0.85)]
    ax.legend(handles=legend_elements, loc='upper left')
    
    # Annotate best and worst
    best_idx = df_sorted['test_both_loss'].idxmin()
    best_val = df_sorted.loc[best_idx, 'test_both_loss']
    ax.axhline(y=best_val, color='green', linestyle='--', alpha=0.5, linewidth=1)
    
    fig.suptitle('All Configurations Ranked by Test Combined Loss (lower is better)', fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '07_all_configs_ranked.png'), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: 07_all_configs_ranked.png")


def plot_metric_comparison_by_horizon(df: pd.DataFrame, output_dir: str):
    """Plot how different metrics behave across prediction horizons, split by model."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for ax, model in zip(axes, ['nn', 'cnn']):
        model_df = df[df['model_type'] == model]
        days_predicted = sorted(model_df['days_predicted'].unique())
        
        for metric, color, label in [('test_bpm_loss', METRIC_COLORS['bpm'], 'BPM (MSE)'),
                                      ('test_both_loss', METRIC_COLORS['both'], 'Combined')]:
            means = model_df.groupby('days_predicted')[metric].mean()
            stds = model_df.groupby('days_predicted')[metric].std()
            ax.errorbar(days_predicted, means, yerr=stds,
                       marker='o', markersize=7, linewidth=2, capsize=4,
                       label=label, color=color)
        
        # Steps on secondary axis due to different scale
        ax2 = ax.twinx()
        means = model_df.groupby('days_predicted')['test_steps_loss'].mean()
        stds = model_df.groupby('days_predicted')['test_steps_loss'].std()
        ax2.errorbar(days_predicted, means, yerr=stds,
                    marker='s', markersize=7, linewidth=2, capsize=4,
                    label='Steps (MAE)', color=METRIC_COLORS['steps'], linestyle='--')
        ax2.set_ylabel('Steps Loss (MAE)', color=METRIC_COLORS['steps'])
        ax2.tick_params(axis='y', labelcolor=METRIC_COLORS['steps'])
        
        ax.set_xlabel('Prediction Horizon (days)')
        ax.set_ylabel('BPM / Combined Loss')
        ax.set_title(f'{model.upper()} Model')
        ax.set_xticks(days_predicted)
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    fig.suptitle('Test Losses by Prediction Horizon (All Metrics)', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '08_metrics_by_horizon.png'), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: 08_metrics_by_horizon.png")


def plot_best_configs_summary(df: pd.DataFrame, output_dir: str):
    """Create a summary plot highlighting best configurations per objective."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    metrics = [
        ('test_bpm_loss', 'Best for BPM Prediction'),
        ('test_steps_loss', 'Best for Steps Prediction'),
        ('test_both_loss', 'Best Overall (Combined)')
    ]
    
    for ax, (metric, title) in zip(axes, metrics):
        # Get top 10 configs for this metric
        top10 = df.nsmallest(10, metric)
        
        labels = [f"M{row['masking_ratio']}%\nG{row['days_given']}d P{row['days_predicted']}d\n{row['model_type'].upper()}"
                  for _, row in top10.iterrows()]
        
        colors = [MODEL_COLORS[row['model_type']] for _, row in top10.iterrows()]
        
        bars = ax.barh(range(len(top10)), top10[metric], color=colors, alpha=0.85, edgecolor='white')
        ax.set_yticks(range(len(top10)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()  # Best on top
        ax.set_xlabel(metric.replace('_', ' ').title())
        ax.set_title(title)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=MODEL_COLORS['nn'], label='NN', alpha=0.85),
                      Patch(facecolor=MODEL_COLORS['cnn'], label='CNN', alpha=0.85)]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.02))
    
    fig.suptitle('Top 10 Configurations per Objective (lower is better)', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '09_best_configs.png'), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: 09_best_configs.png")


def generate_summary_table(df: pd.DataFrame, output_dir: str):
    """Generate a summary statistics table and save as CSV."""
    # Overall summary by model type
    model_summary = df.groupby('model_type').agg({
        'test_bpm_loss': ['mean', 'std', 'min', 'max'],
        'test_steps_loss': ['mean', 'std', 'min', 'max'],
        'test_both_loss': ['mean', 'std', 'min', 'max']
    }).round(2)
    model_summary.to_csv(os.path.join(output_dir, 'summary_by_model.csv'))
    
    # Summary by masking ratio
    masking_summary = df.groupby('masking_ratio').agg({
        'test_bpm_loss': ['mean', 'std'],
        'test_steps_loss': ['mean', 'std'],
        'test_both_loss': ['mean', 'std']
    }).round(2)
    masking_summary.to_csv(os.path.join(output_dir, 'summary_by_masking.csv'))
    
    # Full results table
    df.to_csv(os.path.join(output_dir, 'all_results.csv'), index=False)
    
    print("Saved: summary_by_model.csv, summary_by_masking.csv, all_results.csv")


def main():
    # Paths
    base_dir = '/data1/mjma/ssl-physio/embedding_sequencers_results'
    output_dir = os.path.join('/data1/mjma/ssl-physio', 'plots')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading results from: {base_dir}")
    df = load_all_results(base_dir)
    print(f"Loaded {len(df)} experiment configurations")
    print(f"\nData overview:")
    print(f"  Masking ratios: {sorted(df['masking_ratio'].unique())}")
    print(f"  Days given: {sorted(df['days_given'].unique())}")
    print(f"  Days predicted: {sorted(df['days_predicted'].unique())}")
    print(f"  Model types: {sorted(df['model_type'].unique())}")
    print(f"\nGenerating plots to: {output_dir}")
    print("-" * 50)
    
    # Generate all plots
    plot_model_comparison(df, output_dir)
    plot_masking_effect(df, output_dir)
    plot_prediction_horizon_effect(df, output_dir)
    plot_context_length_effect(df, output_dir)
    plot_heatmaps(df, output_dir)
    plot_train_test_comparison(df, output_dir)
    plot_comprehensive_bar(df, output_dir)
    plot_metric_comparison_by_horizon(df, output_dir)
    plot_best_configs_summary(df, output_dir)
    generate_summary_table(df, output_dir)
    
    print("-" * 50)
    print(f"All plots saved to: {output_dir}")
    
    # Print quick summary
    print("\n=== Quick Summary ===")
    best_overall = df.loc[df['test_both_loss'].idxmin()]
    print(f"\nBest overall config (lowest test_both_loss):")
    print(f"  Masking: {best_overall['masking_ratio']}%")
    print(f"  Days given: {best_overall['days_given']}")
    print(f"  Days predicted: {best_overall['days_predicted']}")
    print(f"  Model: {best_overall['model_type'].upper()}")
    print(f"  Test combined loss: {best_overall['test_both_loss']:.2f}")


if __name__ == '__main__':
    main()


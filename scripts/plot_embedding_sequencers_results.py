#!/usr/bin/env python3
"""
Plotting script for embedding sequencers experiment results.
Generates visualizations comparing performance across different experimental conditions.

Output types:
- bpm: regression (MSE loss)
- steps: regression (MAE loss)
- anxiety: binary classification (BCE loss, metrics: balanced_acc, f1, auc, etc.)
- stress: binary classification (BCE loss, metrics: balanced_acc, f1, auc, etc.)

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
from matplotlib.patches import Patch
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
METRIC_COLORS = {
    'bpm': '#e74c3c', 
    'steps': '#3498db', 
    'anxiety': '#9b59b6',
    'stress': '#27ae60'
}
CLASSIFICATION_METRIC_COLORS = {
    'balanced_accuracy': '#e74c3c',
    'accuracy': '#3498db',
    'f1': '#2ecc71',
    'precision': '#f39c12',
    'recall': '#9b59b6',
    'auc': '#1abc9c'
}


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
                
                # Check if this is the new format (has anxiety/stress) or old format (has both)
                is_new_format = 'test_anxiety_balanced_accuracy' in avg_results
                
                record = {
                    'masking_ratio': int(params['masking_model'].replace('masking_', '')),
                    'days_given': params['days_given'],
                    'days_predicted': params['days_predicted'],
                    'model_type': params['prediction_model'],
                    'train_bpm_loss': avg_results.get('train_bpm_loss', np.nan),
                    'train_steps_loss': avg_results.get('train_steps_loss', np.nan),
                    'test_bpm_loss': avg_results.get('test_bpm_loss', np.nan),
                    'test_steps_loss': avg_results.get('test_steps_loss', np.nan),
                    'num_subjects': data['metadata']['num_subjects'],
                }
                
                # Add new classification metrics if available
                if is_new_format:
                    # Anxiety metrics
                    record['train_anxiety_loss'] = avg_results.get('train_anxiety_loss', np.nan)
                    record['test_anxiety_loss'] = avg_results.get('test_anxiety_loss', np.nan)
                    record['test_anxiety_balanced_accuracy'] = avg_results.get('test_anxiety_balanced_accuracy', np.nan)
                    record['test_anxiety_accuracy'] = avg_results.get('test_anxiety_accuracy', np.nan)
                    record['test_anxiety_f1'] = avg_results.get('test_anxiety_f1', np.nan)
                    record['test_anxiety_precision'] = avg_results.get('test_anxiety_precision', np.nan)
                    record['test_anxiety_recall'] = avg_results.get('test_anxiety_recall', np.nan)
                    record['test_anxiety_auc'] = avg_results.get('test_anxiety_auc', np.nan)
                    
                    # Stress metrics
                    record['train_stress_loss'] = avg_results.get('train_stress_loss', np.nan)
                    record['test_stress_loss'] = avg_results.get('test_stress_loss', np.nan)
                    record['test_stress_balanced_accuracy'] = avg_results.get('test_stress_balanced_accuracy', np.nan)
                    record['test_stress_accuracy'] = avg_results.get('test_stress_accuracy', np.nan)
                    record['test_stress_f1'] = avg_results.get('test_stress_f1', np.nan)
                    record['test_stress_precision'] = avg_results.get('test_stress_precision', np.nan)
                    record['test_stress_recall'] = avg_results.get('test_stress_recall', np.nan)
                    record['test_stress_auc'] = avg_results.get('test_stress_auc', np.nan)
                
                records.append(record)
    
    df = pd.DataFrame(records)
    return df.sort_values(['masking_ratio', 'days_given', 'days_predicted', 'model_type']).reset_index(drop=True)


def plot_regression_model_comparison(df: pd.DataFrame, output_dir: str):
    """Plot NN vs CNN comparison for regression tasks (BPM, Steps)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    metrics = [
        ('test_bpm_loss', 'BPM Loss (MSE)', 'Heart Rate Prediction'),
        ('test_steps_loss', 'Steps Loss (MAE)', 'Step Count Prediction'),
    ]
    
    for ax, (metric, ylabel, title) in zip(axes, metrics):
        # Group by model and configuration, then merge to ensure alignment
        nn_data = df[df['model_type'] == 'nn'].groupby(['masking_ratio', 'days_given'])[metric].mean().reset_index()
        cnn_data = df[df['model_type'] == 'cnn'].groupby(['masking_ratio', 'days_given'])[metric].mean().reset_index()
        
        # Merge to get only common configurations
        merged = pd.merge(nn_data, cnn_data, on=['masking_ratio', 'days_given'], 
                         suffixes=('_nn', '_cnn'), how='outer')
        
        x = np.arange(len(merged))
        width = 0.35
        
        labels = [f"M{row['masking_ratio']}% G{row['days_given']}d" 
                  for _, row in merged.iterrows()]
        
        nn_vals = merged[f'{metric}_nn'].fillna(0)
        cnn_vals = merged[f'{metric}_cnn'].fillna(0)
        
        # Plot bars only where data exists
        nn_mask = ~merged[f'{metric}_nn'].isna()
        cnn_mask = ~merged[f'{metric}_cnn'].isna()
        
        if nn_mask.any():
            ax.bar(x[nn_mask] - width/2, nn_vals[nn_mask], width, label='NN', 
                   color=MODEL_COLORS['nn'], alpha=0.85, edgecolor='white')
        if cnn_mask.any():
            ax.bar(x[cnn_mask] + width/2, cnn_vals[cnn_mask], width, label='CNN', 
                   color=MODEL_COLORS['cnn'], alpha=0.85, edgecolor='white')
        
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
    
    fig.suptitle('Regression Tasks: NN vs CNN Comparison', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_regression_model_comparison.png'), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: 01_regression_model_comparison.png")


def plot_classification_model_comparison(df: pd.DataFrame, output_dir: str):
    """Plot NN vs CNN comparison for classification tasks (Anxiety, Stress)."""
    # Check if we have classification data
    if 'test_anxiety_balanced_accuracy' not in df.columns or df['test_anxiety_balanced_accuracy'].isna().all():
        print("Skipped: 02_classification_model_comparison.png (no classification data)")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    
    targets = ['anxiety', 'stress']
    metric_pairs = [
        ('balanced_accuracy', 'Balanced Accuracy'),
        ('f1', 'F1 Score'),
    ]
    
    for row_idx, target in enumerate(targets):
        for col, (metric_suffix, metric_label) in enumerate(metric_pairs):
            ax = axes[row_idx, col]
            metric = f'test_{target}_{metric_suffix}'
            
            if metric not in df.columns:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                continue
            
            nn_data = df[df['model_type'] == 'nn'].groupby(['masking_ratio', 'days_given'])[metric].mean().reset_index()
            cnn_data = df[df['model_type'] == 'cnn'].groupby(['masking_ratio', 'days_given'])[metric].mean().reset_index()
            
            # Merge to get only common configurations
            merged = pd.merge(nn_data, cnn_data, on=['masking_ratio', 'days_given'], 
                             suffixes=('_nn', '_cnn'), how='outer')
            
            x = np.arange(len(merged))
            width = 0.35
            
            labels = [f"M{row_data['masking_ratio']}% G{row_data['days_given']}d" 
                      for _, row_data in merged.iterrows()]
            
            nn_mask = ~merged[f'{metric}_nn'].isna()
            cnn_mask = ~merged[f'{metric}_cnn'].isna()
            
            if nn_mask.any():
                ax.bar(x[nn_mask] - width/2, merged.loc[nn_mask, f'{metric}_nn'], width, label='NN', 
                       color=MODEL_COLORS['nn'], alpha=0.85, edgecolor='white')
            if cnn_mask.any():
                ax.bar(x[cnn_mask] + width/2, merged.loc[cnn_mask, f'{metric}_cnn'], width, label='CNN', 
                       color=MODEL_COLORS['cnn'], alpha=0.85, edgecolor='white')
            
            ax.set_ylabel(metric_label)
            ax.set_title(f'{target.capitalize()} - {metric_label}')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
            ax.legend(loc='upper right')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
            ax.set_ylim(0, 1)
    
    fig.suptitle('Classification Tasks: NN vs CNN Comparison', fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_classification_model_comparison.png'), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: 02_classification_model_comparison.png")


def plot_masking_effect_regression(df: pd.DataFrame, output_dir: str):
    """Plot effect of masking ratio on regression test losses."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    
    metrics = [
        ('test_bpm_loss', 'BPM Loss (MSE)'),
        ('test_steps_loss', 'Steps Loss (MAE)'),
    ]
    
    masking_ratios = sorted(df['masking_ratio'].unique())
    
    for ax, (metric, ylabel) in zip(axes, metrics):
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
    
    fig.suptitle('Effect of Pre-training Masking Ratio on Regression Loss', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_masking_effect_regression.png'), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: 03_masking_effect_regression.png")


def plot_masking_effect_classification(df: pd.DataFrame, output_dir: str):
    """Plot effect of masking ratio on classification metrics."""
    if 'test_anxiety_balanced_accuracy' not in df.columns or df['test_anxiety_balanced_accuracy'].isna().all():
        print("Skipped: 04_masking_effect_classification.png (no classification data)")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    
    targets = [('anxiety', 'Anxiety'), ('stress', 'Stress')]
    masking_ratios = sorted(df['masking_ratio'].unique())
    
    for ax, (target, target_label) in zip(axes, targets):
        metric = f'test_{target}_balanced_accuracy'
        
        for model in ['nn', 'cnn']:
            model_df = df[df['model_type'] == model]
            means = model_df.groupby('masking_ratio')[metric].mean()
            stds = model_df.groupby('masking_ratio')[metric].std()
            
            ax.errorbar(masking_ratios, means, yerr=stds, 
                       marker='o', markersize=8, linewidth=2, capsize=5,
                       label=model.upper(), color=MODEL_COLORS[model])
        
        ax.set_xlabel('Masking Ratio (%)')
        ax.set_ylabel('Balanced Accuracy')
        ax.set_title(f'{target_label} Classification')
        ax.set_xticks(masking_ratios)
        ax.legend()
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.set_ylim(0.4, 0.8)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    
    fig.suptitle('Effect of Masking Ratio on Classification (Balanced Accuracy)', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_masking_effect_classification.png'), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: 04_masking_effect_classification.png")


def plot_prediction_horizon_effect(df: pd.DataFrame, output_dir: str):
    """Plot effect of prediction horizon (days_predicted) on all metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    
    days_predicted = sorted(df['days_predicted'].unique())
    
    # Top row: Regression
    metrics_regression = [
        ('test_bpm_loss', 'BPM Loss (MSE)', axes[0, 0]),
        ('test_steps_loss', 'Steps Loss (MAE)', axes[0, 1]),
    ]
    
    for metric, ylabel, ax in metrics_regression:
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
    
    # Bottom row: Classification (if available)
    has_classification = 'test_anxiety_balanced_accuracy' in df.columns and not df['test_anxiety_balanced_accuracy'].isna().all()
    
    if has_classification:
        metrics_classification = [
            ('test_anxiety_balanced_accuracy', 'Anxiety Bal. Accuracy', axes[1, 0]),
            ('test_stress_balanced_accuracy', 'Stress Bal. Accuracy', axes[1, 1]),
        ]
        
        for metric, ylabel, ax in metrics_classification:
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
            ax.set_ylim(0.4, 0.8)
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    else:
        for ax in [axes[1, 0], axes[1, 1]]:
            ax.text(0.5, 0.5, 'Classification data not available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_xlabel('Prediction Horizon (days)')
    
    fig.suptitle('Effect of Prediction Horizon (by Masking Ratio)', fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_prediction_horizon.png'), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: 05_prediction_horizon.png")


def plot_context_length_effect(df: pd.DataFrame, output_dir: str):
    """Plot effect of context length (days_given) on all metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    
    days_given = sorted(df['days_given'].unique())
    
    # Top row: Regression
    metrics_regression = [
        ('test_bpm_loss', 'BPM Loss (MSE)', axes[0, 0]),
        ('test_steps_loss', 'Steps Loss (MAE)', axes[0, 1]),
    ]
    
    for metric, ylabel, ax in metrics_regression:
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
    
    # Bottom row: Classification (if available)
    has_classification = 'test_anxiety_balanced_accuracy' in df.columns and not df['test_anxiety_balanced_accuracy'].isna().all()
    
    if has_classification:
        metrics_classification = [
            ('test_anxiety_balanced_accuracy', 'Anxiety Bal. Accuracy', axes[1, 0]),
            ('test_stress_balanced_accuracy', 'Stress Bal. Accuracy', axes[1, 1]),
        ]
        
        for metric, ylabel, ax in metrics_classification:
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
            ax.set_ylim(0.4, 0.8)
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    else:
        for ax in [axes[1, 0], axes[1, 1]]:
            ax.text(0.5, 0.5, 'Classification data not available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_xlabel('Context Length (days)')
    
    fig.suptitle('Effect of Context Window Length (by Masking Ratio)', fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '06_context_length.png'), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: 06_context_length.png")


def plot_heatmaps_regression(df: pd.DataFrame, output_dir: str):
    """Plot heatmaps of test_bpm_loss for days_given x days_predicted."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    
    masking_ratios = sorted(df['masking_ratio'].unique())
    model_types = ['nn', 'cnn']
    
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    vmin = df['test_bpm_loss'].min()
    vmax = df['test_bpm_loss'].max()
    
    for row, model in enumerate(model_types):
        for col, masking in enumerate(masking_ratios):
            ax = axes[row, col]
            
            subset = df[(df['masking_ratio'] == masking) & (df['model_type'] == model)]
            if len(subset) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                continue
                
            pivot = subset.pivot(index='days_given', columns='days_predicted', values='test_bpm_loss')
            pivot = pivot.sort_index()
            pivot = pivot[sorted(pivot.columns)]
            
            sns.heatmap(pivot, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
                       annot=True, fmt='.0f', cbar=col == 2,
                       cbar_kws={'label': 'BPM Loss (MSE)'} if col == 2 else {},
                       linewidths=0.5, linecolor='white')
            
            ax.set_title(f'{model.upper()} - Mask {masking}%')
            ax.set_xlabel('Days Predicted')
            ax.set_ylabel('Days Given')
    
    fig.suptitle('BPM Regression Loss: Context × Horizon × Masking × Model', fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '07_heatmaps_bpm.png'), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: 07_heatmaps_bpm.png")


def plot_heatmaps_classification(df: pd.DataFrame, output_dir: str):
    """Plot heatmaps of classification balanced accuracy for days_given x days_predicted."""
    if 'test_anxiety_balanced_accuracy' not in df.columns or df['test_anxiety_balanced_accuracy'].isna().all():
        print("Skipped: 08_heatmaps_classification.png (no classification data)")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    targets = [('anxiety', 'Anxiety'), ('stress', 'Stress')]
    models = [('nn', 'NN'), ('cnn', 'CNN')]
    
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    
    for row, (target, target_label) in enumerate(targets):
        metric = f'test_{target}_balanced_accuracy'
        vmin = max(0.4, df[metric].min() - 0.05)
        vmax = min(0.8, df[metric].max() + 0.05)
        
        for col, (model, model_label) in enumerate(models):
            ax = axes[row, col]
            
            subset = df[df['model_type'] == model]
            if len(subset) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Aggregate across masking ratios
            agg = subset.groupby(['days_given', 'days_predicted'])[metric].mean().reset_index()
            pivot = agg.pivot(index='days_given', columns='days_predicted', values=metric)
            pivot = pivot.sort_index()
            pivot = pivot[sorted(pivot.columns)]
            
            sns.heatmap(pivot, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
                       annot=True, fmt='.2f', cbar=col == 1,
                       cbar_kws={'label': 'Balanced Accuracy'} if col == 1 else {},
                       linewidths=0.5, linecolor='white')
            
            ax.set_title(f'{target_label} - {model_label} (avg across masking)')
            ax.set_xlabel('Days Predicted')
            ax.set_ylabel('Days Given')
    
    fig.suptitle('Classification Balanced Accuracy: Context × Horizon', fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '08_heatmaps_classification.png'), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: 08_heatmaps_classification.png")


def plot_classification_metrics_detail(df: pd.DataFrame, output_dir: str):
    """Plot detailed classification metrics comparison."""
    if 'test_anxiety_balanced_accuracy' not in df.columns or df['test_anxiety_balanced_accuracy'].isna().all():
        print("Skipped: 09_classification_metrics_detail.png (no classification data)")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    
    metrics = ['balanced_accuracy', 'accuracy', 'f1', 'precision', 'recall', 'auc']
    metric_labels = ['Balanced Accuracy', 'Accuracy', 'F1 Score', 'Precision', 'Recall', 'AUC']
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]
        
        anxiety_metric = f'test_anxiety_{metric}'
        stress_metric = f'test_stress_{metric}'
        
        # Average across all configurations per model
        for model in ['nn', 'cnn']:
            model_df = df[df['model_type'] == model]
            
            anxiety_mean = model_df[anxiety_metric].mean()
            stress_mean = model_df[stress_metric].mean()
            
            x = [0, 1]
            heights = [anxiety_mean, stress_mean]
            
            offset = -0.2 if model == 'nn' else 0.2
            ax.bar([xi + offset for xi in x], heights, width=0.35, 
                   label=model.upper(), color=MODEL_COLORS[model], alpha=0.85, edgecolor='white')
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Anxiety', 'Stress'])
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        if metric in ['balanced_accuracy', 'accuracy', 'f1', 'precision', 'recall', 'auc']:
            ax.set_ylim(0, 1)
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, label='Random')
    
    fig.suptitle('Classification Metrics Comparison: Anxiety vs Stress', fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '09_classification_metrics_detail.png'), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: 09_classification_metrics_detail.png")


def plot_train_test_comparison(df: pd.DataFrame, output_dir: str):
    """Plot train vs test loss for regression tasks."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    
    metrics = [
        ('train_bpm_loss', 'test_bpm_loss', 'BPM Loss'),
        ('train_steps_loss', 'test_steps_loss', 'Steps Loss'),
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
    plt.savefig(os.path.join(output_dir, '10_train_test_comparison.png'), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: 10_train_test_comparison.png")


def plot_best_configs_summary(df: pd.DataFrame, output_dir: str):
    """Create a summary plot highlighting best configurations per objective."""
    has_classification = 'test_anxiety_balanced_accuracy' in df.columns and not df['test_anxiety_balanced_accuracy'].isna().all()
    
    if has_classification:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        metrics = [
            ('test_bpm_loss', 'Best for BPM (lower is better)', False, axes[0, 0]),
            ('test_steps_loss', 'Best for Steps (lower is better)', False, axes[0, 1]),
            ('test_anxiety_balanced_accuracy', 'Best for Anxiety (higher is better)', True, axes[1, 0]),
            ('test_stress_balanced_accuracy', 'Best for Stress (higher is better)', True, axes[1, 1]),
        ]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        metrics = [
            ('test_bpm_loss', 'Best for BPM (lower is better)', False, axes[0]),
            ('test_steps_loss', 'Best for Steps (lower is better)', False, axes[1]),
        ]
    
    for metric, title, higher_better, ax in metrics:
        if metric not in df.columns or df[metric].isna().all():
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Get top 8 configs for this metric
        if higher_better:
            top8 = df.nlargest(8, metric)
        else:
            top8 = df.nsmallest(8, metric)
        
        labels = [f"M{row['masking_ratio']}%\nG{row['days_given']}d P{row['days_predicted']}d\n{row['model_type'].upper()}"
                  for _, row in top8.iterrows()]
        
        colors = [MODEL_COLORS[row['model_type']] for _, row in top8.iterrows()]
        
        ax.barh(range(len(top8)), top8[metric], color=colors, alpha=0.85, edgecolor='white')
        ax.set_yticks(range(len(top8)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(metric.replace('_', ' ').replace('test ', '').title())
        ax.set_title(title)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
    
    # Add legend
    legend_elements = [Patch(facecolor=MODEL_COLORS['nn'], label='NN', alpha=0.85),
                      Patch(facecolor=MODEL_COLORS['cnn'], label='CNN', alpha=0.85)]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.02))
    
    fig.suptitle('Top 8 Configurations per Objective', fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '11_best_configs.png'), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: 11_best_configs.png")


def plot_all_configs_ranked(df: pd.DataFrame, output_dir: str):
    """Plot all configurations ranked by BPM loss."""
    fig, ax = plt.subplots(figsize=(16, 6))
    
    df_sorted = df.sort_values('test_bpm_loss')
    
    labels = [f"M{row['masking_ratio']}% G{row['days_given']}d P{row['days_predicted']}d {row['model_type'].upper()}"
              for _, row in df_sorted.iterrows()]
    
    x = np.arange(len(labels))
    colors = [MODEL_COLORS[row['model_type']] for _, row in df_sorted.iterrows()]
    
    ax.bar(x, df_sorted['test_bpm_loss'], color=colors, alpha=0.85, edgecolor='white')
    
    ax.set_ylabel('Test BPM Loss (MSE)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, ha='center', fontsize=7)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    legend_elements = [Patch(facecolor=MODEL_COLORS['nn'], label='NN', alpha=0.85),
                      Patch(facecolor=MODEL_COLORS['cnn'], label='CNN', alpha=0.85)]
    ax.legend(handles=legend_elements, loc='upper left')
    
    fig.suptitle('All Configurations Ranked by BPM Loss (lower is better)', fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '12_all_configs_ranked.png'), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved: 12_all_configs_ranked.png")


def generate_summary_table(df: pd.DataFrame, output_dir: str):
    """Generate summary statistics tables and save as CSV."""
    has_classification = 'test_anxiety_balanced_accuracy' in df.columns and not df['test_anxiety_balanced_accuracy'].isna().all()
    
    # Regression summary by model type
    regression_metrics = ['test_bpm_loss', 'test_steps_loss']
    model_summary = df.groupby('model_type')[regression_metrics].agg(['mean', 'std', 'min', 'max']).round(2)
    model_summary.to_csv(os.path.join(output_dir, 'summary_regression_by_model.csv'))
    
    # Regression summary by masking ratio
    masking_summary = df.groupby('masking_ratio')[regression_metrics].agg(['mean', 'std']).round(2)
    masking_summary.to_csv(os.path.join(output_dir, 'summary_regression_by_masking.csv'))
    
    # Classification summary if available
    if has_classification:
        classification_metrics = [
            'test_anxiety_balanced_accuracy', 'test_anxiety_f1', 'test_anxiety_auc',
            'test_stress_balanced_accuracy', 'test_stress_f1', 'test_stress_auc'
        ]
        
        class_model_summary = df.groupby('model_type')[classification_metrics].agg(['mean', 'std']).round(3)
        class_model_summary.to_csv(os.path.join(output_dir, 'summary_classification_by_model.csv'))
        
        class_masking_summary = df.groupby('masking_ratio')[classification_metrics].agg(['mean', 'std']).round(3)
        class_masking_summary.to_csv(os.path.join(output_dir, 'summary_classification_by_masking.csv'))
    
    # Full results table
    df.to_csv(os.path.join(output_dir, 'all_results.csv'), index=False)
    
    print("Saved: summary CSV files and all_results.csv")


def main():
    # Paths
    base_dir = '/data1/mjma/ssl-physio/embedding_sequencers_results'
    output_dir = os.path.join('/data1/mjma/ssl-physio', 'plots')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading results from: {base_dir}")
    df = load_all_results(base_dir)
    print(f"Loaded {len(df)} experiment configurations")
    
    if len(df) == 0:
        print("No results found!")
        return
    
    print(f"\nData overview:")
    print(f"  Masking ratios: {sorted(df['masking_ratio'].unique())}")
    print(f"  Days given: {sorted(df['days_given'].unique())}")
    print(f"  Days predicted: {sorted(df['days_predicted'].unique())}")
    print(f"  Model types: {sorted(df['model_type'].unique())}")
    
    has_classification = 'test_anxiety_balanced_accuracy' in df.columns and not df['test_anxiety_balanced_accuracy'].isna().all()
    print(f"  Classification data available: {has_classification}")
    
    print(f"\nGenerating plots to: {output_dir}")
    print("-" * 50)
    
    # Generate all plots
    plot_regression_model_comparison(df, output_dir)
    plot_classification_model_comparison(df, output_dir)
    plot_masking_effect_regression(df, output_dir)
    plot_masking_effect_classification(df, output_dir)
    plot_prediction_horizon_effect(df, output_dir)
    plot_context_length_effect(df, output_dir)
    plot_heatmaps_regression(df, output_dir)
    plot_heatmaps_classification(df, output_dir)
    plot_classification_metrics_detail(df, output_dir)
    plot_train_test_comparison(df, output_dir)
    plot_best_configs_summary(df, output_dir)
    plot_all_configs_ranked(df, output_dir)
    generate_summary_table(df, output_dir)
    
    print("-" * 50)
    print(f"All plots saved to: {output_dir}")
    
    # Print quick summary
    print("\n=== Quick Summary ===")
    
    best_bpm = df.loc[df['test_bpm_loss'].idxmin()]
    print(f"\nBest BPM config (lowest test_bpm_loss):")
    print(f"  Masking: {best_bpm['masking_ratio']}%")
    print(f"  Days given: {best_bpm['days_given']}")
    print(f"  Days predicted: {best_bpm['days_predicted']}")
    print(f"  Model: {best_bpm['model_type'].upper()}")
    print(f"  Test BPM loss (MSE): {best_bpm['test_bpm_loss']:.2f}")
    
    if has_classification:
        best_anxiety = df.loc[df['test_anxiety_balanced_accuracy'].idxmax()]
        print(f"\nBest Anxiety config (highest balanced accuracy):")
        print(f"  Masking: {best_anxiety['masking_ratio']}%")
        print(f"  Days given: {best_anxiety['days_given']}")
        print(f"  Days predicted: {best_anxiety['days_predicted']}")
        print(f"  Model: {best_anxiety['model_type'].upper()}")
        print(f"  Balanced accuracy: {best_anxiety['test_anxiety_balanced_accuracy']:.3f}")
        
        best_stress = df.loc[df['test_stress_balanced_accuracy'].idxmax()]
        print(f"\nBest Stress config (highest balanced accuracy):")
        print(f"  Masking: {best_stress['masking_ratio']}%")
        print(f"  Days given: {best_stress['days_given']}")
        print(f"  Days predicted: {best_stress['days_predicted']}")
        print(f"  Model: {best_stress['model_type'].upper()}")
        print(f"  Balanced accuracy: {best_stress['test_stress_balanced_accuracy']:.3f}")


if __name__ == '__main__':
    main()

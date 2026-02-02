"""
Analyze Seed Variance

Measures how random weight initialization (different seeds) affects test metrics
for each model configuration. This answers: "With the same data and model setup,
how much does randomness in initialization cause variance in results?"

Each configuration was run with 5 seeds (0-4). This script summarizes the
variance across those seeds to understand model stability/reliability.

Output: embedding_sequencers_results/seed_variance_summary.json
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np


# Test metrics to analyze (only test metrics, not train)
TEST_METRICS = {
    'regression': ['test_bpm_loss', 'test_steps_loss'],
    'classification': [
        'test_anxiety_balanced_accuracy', 'test_anxiety_f1', 'test_anxiety_auc',
        'test_stress_balanced_accuracy', 'test_stress_f1', 'test_stress_auc'
    ]
}

ALL_TEST_METRICS = TEST_METRICS['regression'] + TEST_METRICS['classification']


def load_all_seed_results(base_dir: str) -> list:
    """
    Load all seed_results.json files from the results directory.
    
    Returns:
        List of dicts with config info and aggregated results
    """
    results = []
    base_path = Path(base_dir)
    
    for seed_file in base_path.rglob('seed_results.json'):
        with open(seed_file, 'r') as f:
            data = json.load(f)
        
        params = data['parameters']
        config = {
            'masking_model': params['masking_model'],
            'days_given': params['days_given'],
            'days_predicted': params['days_predicted'],
            'prediction_model': params['prediction_model'],
            'path': str(seed_file.relative_to(base_path)),
            'aggregated_results': data['aggregated_results']
        }
        results.append(config)
    
    return results


def compute_cv(mean: float, std: float) -> float:
    """Compute coefficient of variation (std/mean), handling edge cases."""
    if mean == 0 or np.isnan(mean) or np.isnan(std):
        return float('nan')
    return abs(std / mean)


def extract_metric_stats(results: list, metric: str) -> dict:
    """
    Extract mean, std, and CV for a metric across all configs.
    
    Returns dict with lists of values for each stat.
    """
    means = []
    stds = []
    cvs = []
    
    for r in results:
        agg = r['aggregated_results']
        mean_key = f'{metric}_mean'
        std_key = f'{metric}_std'
        
        if mean_key in agg and std_key in agg:
            mean_val = agg[mean_key]
            std_val = agg[std_key]
            cv_val = compute_cv(mean_val, std_val)
            
            if not np.isnan(mean_val):
                means.append(mean_val)
            if not np.isnan(std_val):
                stds.append(std_val)
            if not np.isnan(cv_val):
                cvs.append(cv_val)
    
    return {'means': means, 'stds': stds, 'cvs': cvs}


def summarize_overall(results: list) -> dict:
    """
    Compute overall variance statistics across all configurations.
    """
    summary = {}
    
    for metric in ALL_TEST_METRICS:
        stats = extract_metric_stats(results, metric)
        
        if stats['stds']:
            summary[metric] = {
                'avg_std': float(np.mean(stats['stds'])),
                'avg_cv': float(np.mean(stats['cvs'])) if stats['cvs'] else None,
                'max_std': float(np.max(stats['stds'])),
                'min_std': float(np.min(stats['stds'])),
                'avg_mean': float(np.mean(stats['means'])) if stats['means'] else None,
            }
    
    return summary


def summarize_by_slice(results: list, slice_key: str) -> dict:
    """
    Compute variance statistics grouped by a config key (e.g., prediction_model).
    """
    # Group results by the slice key
    groups = defaultdict(list)
    for r in results:
        key_val = r[slice_key]
        groups[key_val].append(r)
    
    summary = {}
    for group_val, group_results in sorted(groups.items()):
        group_summary = {}
        
        for metric in ALL_TEST_METRICS:
            stats = extract_metric_stats(group_results, metric)
            
            if stats['stds']:
                group_summary[metric] = {
                    'avg_std': float(np.mean(stats['stds'])),
                    'avg_cv': float(np.mean(stats['cvs'])) if stats['cvs'] else None,
                    'n_configs': len(stats['stds'])
                }
        
        summary[str(group_val)] = group_summary
    
    return summary


def find_extreme_variance_configs(results: list, n: int = 3) -> dict:
    """
    Find configurations with highest and lowest variance for key metrics.
    """
    extremes = {}
    
    # Focus on key representative metrics
    key_metrics = [
        'test_bpm_loss',
        'test_anxiety_balanced_accuracy', 
        'test_stress_balanced_accuracy'
    ]
    
    for metric in key_metrics:
        # Collect (config_name, std, cv) tuples
        config_variances = []
        
        for r in results:
            agg = r['aggregated_results']
            std_key = f'{metric}_std'
            mean_key = f'{metric}_mean'
            
            if std_key in agg and mean_key in agg:
                std_val = agg[std_key]
                mean_val = agg[mean_key]
                cv_val = compute_cv(mean_val, std_val)
                
                if not np.isnan(std_val):
                    config_name = f"{r['masking_model']}/{r['days_given']}d_in/{r['days_predicted']}d_out/{r['prediction_model']}"
                    config_variances.append({
                        'config': config_name,
                        'std': std_val,
                        'cv': cv_val,
                        'mean': mean_val
                    })
        
        # Sort by CV (normalized variance) for fair comparison
        sorted_by_cv = sorted(config_variances, key=lambda x: x['cv'] if not np.isnan(x['cv']) else float('inf'))
        
        extremes[metric] = {
            'most_stable': sorted_by_cv[:n],
            'least_stable': sorted_by_cv[-n:][::-1]
        }
    
    return extremes


def compute_variance_interpretation(overall_summary: dict) -> dict:
    """
    Provide human-readable interpretation of variance levels.
    """
    interpretations = {}
    
    for metric, stats in overall_summary.items():
        cv = stats.get('avg_cv')
        if cv is None:
            interpretation = "unknown"
        elif cv < 0.01:
            interpretation = "very low variance (< 1% CV)"
        elif cv < 0.05:
            interpretation = "low variance (1-5% CV)"
        elif cv < 0.10:
            interpretation = "moderate variance (5-10% CV)"
        elif cv < 0.20:
            interpretation = "high variance (10-20% CV)"
        else:
            interpretation = "very high variance (> 20% CV)"
        
        interpretations[metric] = {
            'cv_percent': f"{cv * 100:.2f}%" if cv else "N/A",
            'interpretation': interpretation
        }
    
    return interpretations


def main():
    # Paths
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / 'embedding_sequencers_results'
    output_path = results_dir / 'seed_variance_summary.json'
    
    print(f"Loading seed results from: {results_dir}")
    
    # Load all results
    results = load_all_seed_results(results_dir)
    print(f"Found {len(results)} configurations with seed results")
    
    if not results:
        print("No seed_results.json files found!")
        return
    
    # Build summary
    summary = {
        'description': (
            'Seed variance analysis: How much does random weight initialization '
            'affect test metrics? Each config was run with 5 seeds (0-4) on the '
            'same data. CV (coefficient of variation) = std/mean indicates relative variance.'
        ),
        'num_configurations': len(results),
        'num_seeds_per_config': 5,
        'metrics_analyzed': ALL_TEST_METRICS,
        
        # 1. Overall: average seed-induced variance across all configs
        'overall': summarize_overall(results),
        
        # 2. Human-readable interpretations
        'interpretations': compute_variance_interpretation(summarize_overall(results)),
        
        # 3. Does CNN or NN have more seed variance?
        'by_prediction_model': summarize_by_slice(results, 'prediction_model'),
        
        # 4. Does masking ratio affect seed variance?
        'by_masking_model': summarize_by_slice(results, 'masking_model'),
        
        # 5. Does prediction horizon affect seed variance?
        'by_days_predicted': summarize_by_slice(results, 'days_predicted'),
        
        # 6. Does context length affect seed variance?
        'by_days_given': summarize_by_slice(results, 'days_given'),
        
        # 7. Which specific configs are most/least sensitive to random init?
        'extreme_configs': find_extreme_variance_configs(results),
    }
    
    # Save
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSaved variance summary to: {output_path}")
    
    # Print quick summary
    print("\n" + "="*70)
    print("SEED VARIANCE SUMMARY")
    print("How much does random weight initialization affect test metrics?")
    print("="*70)
    
    print("\n--- Regression Metrics (seed-induced CV) ---")
    for metric in TEST_METRICS['regression']:
        interp = summary['interpretations'].get(metric, {})
        print(f"  {metric}: {interp.get('cv_percent', 'N/A')} - {interp.get('interpretation', 'unknown')}")
    
    print("\n--- Classification Metrics (seed-induced CV) ---")
    for metric in TEST_METRICS['classification']:
        interp = summary['interpretations'].get(metric, {})
        print(f"  {metric}: {interp.get('cv_percent', 'N/A')} - {interp.get('interpretation', 'unknown')}")
    
    print("\n--- Which model type is more sensitive to random init? ---")
    for model, model_stats in summary['by_prediction_model'].items():
        cvs = [v['avg_cv'] for v in model_stats.values() if v.get('avg_cv')]
        avg_cv = np.mean(cvs) if cvs else float('nan')
        print(f"  {model.upper()}: {avg_cv*100:.2f}% average CV across metrics")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()

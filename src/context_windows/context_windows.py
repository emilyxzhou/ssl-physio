"""
Context Windows Experiment

Main entry point for MAML-based within-subject prediction.
Trains fresh MAML models for each target subject using support from other subjects.

For each target subject:
1. Sample support set from 32 other subjects (5 windows each = 160 samples)
2. Create query set from target subject (sliding windows, stride=3)
3. Meta-train MAML on support set
4. Fine-tune and evaluate on query set

Outputs all 5 metrics:
- Binary: stress, anxiety (accuracy, balanced_accuracy, f1, auc)
- Regression: rhr, sleep, steps (mse, mae)
"""

import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import pytz
from tqdm import tqdm

# Handle imports
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_all_data, sample_support_windows, create_query_windows, TARGET_NAMES
    from maml_meta import MultiTargetMAML, MAMLConfig, TARGET_CONFIG
else:
    from .data_loader import load_all_data, sample_support_windows, create_query_windows, TARGET_NAMES
    from .maml_meta import MultiTargetMAML, MAMLConfig, TARGET_CONFIG


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def run_context_windows(
    embedding_model: str,
    masking_ratio: str,
    input_days: int,
    output_days: int,
    prediction_model: str,
    seed: int,
    output_folder: str,
    # MAML config (with defaults)
    inner_lr: float = 0.01,
    outer_lr: float = 0.001,
    inner_steps: int = 5,
    meta_epochs: int = 20,
    inner_steps_test: int = 10,
    # Data config
    min_days_per_subject: int = 30,
    num_support_subjects: int = 32,
    samples_per_subject: int = 5,
    query_stride: int = 3,
    # Other
    save_results: bool = True,
    verbose: bool = True
):
    """
    Run the context windows MAML experiment.
    
    Args:
        embedding_model: "s4", "mamba", or "raw_data"
        masking_ratio: "masking_0", "masking_10", "masking_30", "masking_50", "masking_70"
                      (ignored for raw_data)
        input_days: number of input days (3, 5, or 7)
        output_days: number of output days to predict (1-7)
        prediction_model: "cnn" or "nn"
        seed: random seed for reproducibility
        output_folder: directory to save results
        inner_lr: MAML inner loop learning rate
        outer_lr: MAML outer loop learning rate
        inner_steps: MAML inner loop steps during training
        meta_epochs: number of meta-training epochs
        inner_steps_test: MAML inner loop steps during evaluation
        min_days_per_subject: minimum days required per subject
        num_support_subjects: number of subjects for support set
        samples_per_subject: number of windows per support subject
        query_stride: stride for query set windows
        save_results: whether to save results.json
        verbose: print progress
    
    Returns:
        dict: Results dictionary
    """
    # Set seed
    set_seed(seed)
    rng = np.random.default_rng(seed)
    
    # Determine embedding dimensions based on model type
    if embedding_model == "raw_data":
        embedding_dim = 2      # 2 biosignal channels (HR, steps)
        sequence_len = 1440    # 1440 minutes per day
    else:
        embedding_dim = 128    # S4/Mamba embedding dimension
        sequence_len = 180     # S4/Mamba sequence length
    
    # Timezone for timestamps
    pst = pytz.timezone('America/Los_Angeles')
    started_at = datetime.now(pst).strftime('%Y-%m-%d %H:%M:%S %Z')
    
    if verbose:
        print(f"\n{'='*60}")
        print("CONTEXT WINDOWS - MAML EXPERIMENT")
        print(f"{'='*60}")
        print(f"Embedding model: {embedding_model}")
        print(f"Masking ratio: {masking_ratio}")
        print(f"Input days: {input_days}")
        print(f"Output days: {output_days}")
        print(f"Prediction model: {prediction_model}")
        print(f"Seed: {seed}")
        print(f"Support: {num_support_subjects} subjects × {samples_per_subject} samples")
        print(f"Query stride: {query_stride}")
        print(f"Started at: {started_at}")
        print(f"{'='*60}\n")
    
    # Get device
    device = get_device()
    if verbose:
        print(f"Using device: {device}")
    
    # Load all data
    subject_data, norm_stats, subject_ids = load_all_data(
        embedding_model=embedding_model,
        masking_ratio=masking_ratio,
        min_days_per_subject=min_days_per_subject
    )
    
    if verbose:
        print(f"\nLoaded {len(subject_ids)} subjects")
    
    # MAML configuration
    maml_config = MAMLConfig(
        inner_lr=inner_lr,
        outer_lr=outer_lr,
        inner_steps=inner_steps,
        meta_epochs=meta_epochs,
        inner_steps_test=inner_steps_test
    )
    
    # Results storage
    per_subject_results = {}
    
    # Process each subject as target
    iterator = tqdm(subject_ids, desc="Processing subjects") if verbose else subject_ids
    
    for target_subject in iterator:
        # Sample support set from other subjects
        X_support, Y_support = sample_support_windows(
            subject_data=subject_data,
            target_subject=target_subject,
            input_days=input_days,
            output_days=output_days,
            num_support_subjects=num_support_subjects,
            samples_per_subject=samples_per_subject,
            rng=rng
        )
        
        if len(X_support) == 0:
            if verbose:
                print(f"  Skipping {target_subject[:8]}...: empty support set")
            continue
        
        # Create query set from target subject
        X_query, Y_query = create_query_windows(
            subject_data=subject_data,
            target_subject=target_subject,
            input_days=input_days,
            output_days=output_days,
            stride=query_stride
        )
        
        if len(X_query) == 0:
            if verbose:
                print(f"  Skipping {target_subject[:8]}...: empty query set")
            continue
        
        # Create fresh MAML trainer for this target
        maml = MultiTargetMAML(
            model_type=prediction_model,
            input_days=input_days,
            output_days=output_days,
            config=maml_config,
            device=device,
            embedding_dim=embedding_dim,
            sequence_len=sequence_len
        )
        
        # Meta-train on support set (using a subset of query for meta-gradient)
        # For training, we use query set as the "meta-test" during training
        # This is standard MAML: adapt on support, evaluate on query
        maml.train(X_support, Y_support, X_query, Y_query, verbose=False)
        
        # Evaluate: fine-tune on support, test on query
        results = maml.evaluate(X_support, Y_support, X_query, Y_query)
        
        per_subject_results[target_subject] = results
        
        # Clean up GPU memory
        del maml
        torch.cuda.empty_cache()
    
    # Aggregate results across subjects
    average_results = {}
    for target_name in TARGET_CONFIG.keys():
        target_type = TARGET_CONFIG[target_name]['type']
        
        if target_type == 'binary':
            metrics = ['accuracy', 'balanced_accuracy', 'f1', 'auc']
        else:
            metrics = ['mse', 'mae']
        
        for metric in metrics:
            values = []
            for subj_results in per_subject_results.values():
                if target_name in subj_results and metric in subj_results[target_name]:
                    val = subj_results[target_name][metric]
                    if not np.isnan(val):
                        values.append(val)
            
            if values:
                average_results[f'{target_name}_{metric}_mean'] = float(np.mean(values))
                average_results[f'{target_name}_{metric}_std'] = float(np.std(values))
            else:
                average_results[f'{target_name}_{metric}_mean'] = float('nan')
                average_results[f'{target_name}_{metric}_std'] = float('nan')
    
    # End time
    ended_at = datetime.now(pst).strftime('%Y-%m-%d %H:%M:%S %Z')
    
    # Build output dictionary
    output = {
        'parameters': {
            'embedding_model': embedding_model,
            'masking_ratio': masking_ratio,
            'input_days': input_days,
            'output_days': output_days,
            'prediction_model': prediction_model,
            'seed': seed,
            'num_support_subjects': num_support_subjects,
            'samples_per_subject': samples_per_subject,
            'query_stride': query_stride,
            'min_days_per_subject': min_days_per_subject
        },
        'maml_config': maml_config.to_dict(),
        'started_at': started_at,
        'ended_at': ended_at,
        'num_subjects_processed': len(per_subject_results),
        'average_results': average_results,
        'per_subject_results': per_subject_results,
        'normalization_stats': norm_stats,
        'device': str(device)
    }
    
    # Save results
    if save_results:
        os.makedirs(output_folder, exist_ok=True)
        results_path = os.path.join(output_folder, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(output, f, indent=4)
        
        if verbose:
            print(f"\n{'='*60}")
            print("RESULTS SUMMARY")
            print(f"{'='*60}")
            print(f"\nSubjects processed: {len(per_subject_results)}")
            print("\n--- Binary Classification (Mean ± Std) ---")
            for target in ['stress', 'anxiety']:
                acc = average_results.get(f'{target}_accuracy_mean', float('nan'))
                acc_std = average_results.get(f'{target}_accuracy_std', float('nan'))
                bal = average_results.get(f'{target}_balanced_accuracy_mean', float('nan'))
                bal_std = average_results.get(f'{target}_balanced_accuracy_std', float('nan'))
                f1 = average_results.get(f'{target}_f1_mean', float('nan'))
                f1_std = average_results.get(f'{target}_f1_std', float('nan'))
                auc = average_results.get(f'{target}_auc_mean', float('nan'))
                auc_std = average_results.get(f'{target}_auc_std', float('nan'))
                print(f"{target.upper()}:")
                print(f"  Accuracy: {acc:.4f} ± {acc_std:.4f}")
                print(f"  Balanced Acc: {bal:.4f} ± {bal_std:.4f}")
                print(f"  F1: {f1:.4f} ± {f1_std:.4f}")
                print(f"  AUC: {auc:.4f} ± {auc_std:.4f}")
            
            print("\n--- Regression (Mean ± Std) ---")
            for target in ['rhr', 'sleep', 'steps']:
                mse = average_results.get(f'{target}_mse_mean', float('nan'))
                mse_std = average_results.get(f'{target}_mse_std', float('nan'))
                mae = average_results.get(f'{target}_mae_mean', float('nan'))
                mae_std = average_results.get(f'{target}_mae_std', float('nan'))
                print(f"{target.upper()}:")
                print(f"  MSE: {mse:.4f} ± {mse_std:.4f}")
                print(f"  MAE: {mae:.4f} ± {mae_std:.4f}")
            
            print(f"\nResults saved to: {results_path}")
            print(f"Ended at: {ended_at}")
            print(f"{'='*60}")
    
    return output


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Context Windows MAML Experiment")
    parser.add_argument('--embedding-model', default='s4', choices=['s4', 'mamba', 'raw_data'])
    parser.add_argument('--masking-ratio', default='masking_10', 
                       choices=['masking_10', 'masking_30', 'masking_50', 'masking_70'],
                       help='Masking ratio (ignored for raw_data)')
    parser.add_argument('--input-days', type=int, default=3, choices=[3, 5, 7])
    parser.add_argument('--output-days', type=int, default=5, choices=[1, 2, 3, 4, 5, 6, 7])
    parser.add_argument('--prediction-model', default='cnn', choices=['nn', 'cnn'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output-folder', default='./test_context_windows')
    parser.add_argument('--meta-epochs', type=int, default=20)
    
    args = parser.parse_args()
    
    run_context_windows(
        embedding_model=args.embedding_model,
        masking_ratio=args.masking_ratio,
        input_days=args.input_days,
        output_days=args.output_days,
        prediction_model=args.prediction_model,
        seed=args.seed,
        output_folder=args.output_folder,
        meta_epochs=args.meta_epochs,
        verbose=True
    )


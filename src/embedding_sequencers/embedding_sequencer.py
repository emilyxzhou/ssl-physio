"""
Embedding Sequencer

Trains sequence prediction models on pre-computed embeddings.
Per-subject models predicting future biosignals from embedding sequences.

Output types:
- bpm: regression (MAE loss)
- steps: regression (MAE loss)
- anxiety: binary classification (BCE loss)
- stress: binary classification (BCE loss)
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Handle imports whether run as module or directly
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_all_data
    from models import create_model, get_loss_function, OUTPUT_TYPE_CONFIG
else:
    from .data_loader import load_all_data
    from .models import create_model, get_loss_function, OUTPUT_TYPE_CONFIG

import pytz
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    balanced_accuracy_score, accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score
)
from tqdm import tqdm

# All output types to train
OUTPUT_TYPES = ['bpm', 'steps', 'anxiety', 'stress']


# Training hyperparameters
TRAINING_CONFIG = {
    'epochs': 150,
    'batch_size': 16,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'early_stopping_patience': 20,
    'min_delta': 1e-6
}


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def train_model(model, train_X, train_Y, output_type, stats, device, config=None):
    """
    Train a model on the given data.
    
    Args:
        model: EmbeddingSequenceModel instance
        train_X: np.ndarray (num_samples, days_given, 128)
        train_Y: np.ndarray (num_samples, days_predicted, 4) [bpm, steps, anxiety, stress]
        output_type: 'bpm', 'steps', 'anxiety', or 'stress'
        stats: global statistics dict
        device: torch device
        config: training config dict (uses TRAINING_CONFIG if None)
    
    Returns:
        final_loss: final training loss
    """
    if config is None:
        config = TRAINING_CONFIG
    
    model = model.to(device)
    model.train()
    
    # Get target index from config
    target_idx = OUTPUT_TYPE_CONFIG[output_type]['target_idx']
    targets = train_Y[:, :, target_idx:target_idx+1]  # (N, days_predicted, 1)
    
    # Convert to tensors
    X_tensor = torch.tensor(train_X, dtype=torch.float32).to(device)
    Y_tensor = torch.tensor(targets, dtype=torch.float32).to(device)
    
    # Create data loader
    dataset = TensorDataset(X_tensor, Y_tensor)
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    # Loss and optimizer
    loss_fn = get_loss_function(output_type)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], 
                          weight_decay=config['weight_decay'])
    
    # Training loop with early stopping
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_X, batch_Y in loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = loss_fn(pred, batch_Y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        
        # Early stopping check
        if avg_loss < best_loss - config['min_delta']:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping_patience']:
                break
    
    return best_loss


def evaluate_model(model, test_X, test_Y, output_type, stats, device):
    """
    Evaluate a model on test data.
    
    Args:
        model: trained EmbeddingSequenceModel
        test_X: np.ndarray (1, days_given, 128)
        test_Y: np.ndarray (1, days_predicted, 4) [bpm, steps, anxiety, stress]
        output_type: 'bpm', 'steps', 'anxiety', or 'stress'
        stats: global statistics dict
        device: torch device
    
    Returns:
        For regression (bpm, steps): dict with 'loss' (MSE or MAE)
        For classification (anxiety, stress): dict with classification metrics
    """
    model.eval()
    
    # Get target index
    target_idx = OUTPUT_TYPE_CONFIG[output_type]['target_idx']
    task_type = OUTPUT_TYPE_CONFIG[output_type]['task']
    targets = test_Y[:, :, target_idx:target_idx+1]  # (1, days_predicted, 1)
    
    X_tensor = torch.tensor(test_X, dtype=torch.float32).to(device)
    Y_tensor = torch.tensor(targets, dtype=torch.float32).to(device)
    
    loss_fn = get_loss_function(output_type)
    
    with torch.no_grad():
        pred = model(X_tensor)  # (1, days_predicted, 1)
        loss = loss_fn(pred, Y_tensor)
    
    if task_type == 'regression':
        return {'loss': loss.item()}
    else:
        # Binary classification - compute additional metrics
        pred_np = pred.cpu().numpy().flatten()  # logits
        target_np = targets.flatten()
        
        # Filter out NaN targets
        valid_mask = ~np.isnan(target_np)
        if valid_mask.sum() == 0:
            # No valid labels - return NaN for all metrics
            return {
                'loss': float('nan'),
                'balanced_accuracy': float('nan'),
                'accuracy': float('nan'),
                'f1': float('nan'),
                'precision': float('nan'),
                'recall': float('nan'),
                'auc': float('nan'),
                'n_valid': 0
            }
        
        pred_valid = pred_np[valid_mask]
        target_valid = target_np[valid_mask].astype(int)
        
        # Convert logits to probabilities and predictions
        pred_probs = 1 / (1 + np.exp(-pred_valid))  # sigmoid
        pred_labels = (pred_probs >= 0.5).astype(int)
        
        # Compute metrics
        metrics = {
            'loss': loss.item(),
            'n_valid': int(valid_mask.sum())
        }
        
        try:
            metrics['balanced_accuracy'] = balanced_accuracy_score(target_valid, pred_labels)
        except:
            metrics['balanced_accuracy'] = float('nan')
        
        try:
            metrics['accuracy'] = accuracy_score(target_valid, pred_labels)
        except:
            metrics['accuracy'] = float('nan')
        
        try:
            metrics['f1'] = f1_score(target_valid, pred_labels, zero_division=0)
        except:
            metrics['f1'] = float('nan')
        
        try:
            metrics['precision'] = precision_score(target_valid, pred_labels, zero_division=0)
        except:
            metrics['precision'] = float('nan')
        
        try:
            metrics['recall'] = recall_score(target_valid, pred_labels, zero_division=0)
        except:
            metrics['recall'] = float('nan')
        
        try:
            # AUC requires both classes present
            if len(np.unique(target_valid)) > 1:
                metrics['auc'] = roc_auc_score(target_valid, pred_probs)
            else:
                metrics['auc'] = float('nan')
        except:
            metrics['auc'] = float('nan')
        
        return metrics


def train_and_evaluate_subject(user_id, data, model_type, days_given, days_predicted, 
                                stats, device, verbose=False):
    """
    Train and evaluate all four output types for a single subject.
    
    Returns:
        results: dict with train/test losses and metrics for bpm, steps, anxiety, stress
        hyperparams: model hyperparameters (from first model created)
    """
    train_X = data['train_X']
    train_Y = data['train_Y']
    test_X = data['test_X']
    test_Y = data['test_Y']
    
    results = {}
    hyperparams = None
    
    for output_type in OUTPUT_TYPES:
        # Create fresh model
        model = create_model(model_type, days_given, days_predicted, output_type)
        
        if hyperparams is None:
            hyperparams = model.get_hyperparameters()
        
        task_type = OUTPUT_TYPE_CONFIG[output_type]['task']
        
        # Train
        train_loss = train_model(model, train_X, train_Y, output_type, stats, device)
        
        # Evaluate
        test_metrics = evaluate_model(model, test_X, test_Y, output_type, stats, device)
        
        # Store results
        results[f'train_{output_type}_loss'] = train_loss
        
        if task_type == 'regression':
            results[f'test_{output_type}_loss'] = test_metrics['loss']
            if verbose:
                print(f"    {output_type}: train_loss={train_loss:.4f}, test_loss={test_metrics['loss']:.4f}")
        else:
            # Binary classification - store all metrics
            results[f'test_{output_type}_loss'] = test_metrics['loss']
            results[f'test_{output_type}_balanced_accuracy'] = test_metrics['balanced_accuracy']
            results[f'test_{output_type}_accuracy'] = test_metrics['accuracy']
            results[f'test_{output_type}_f1'] = test_metrics['f1']
            results[f'test_{output_type}_precision'] = test_metrics['precision']
            results[f'test_{output_type}_recall'] = test_metrics['recall']
            results[f'test_{output_type}_auc'] = test_metrics['auc']
            results[f'test_{output_type}_n_valid'] = test_metrics['n_valid']
            
            if verbose:
                print(f"    {output_type}: train_loss={train_loss:.4f}, "
                      f"test_bal_acc={test_metrics['balanced_accuracy']:.4f}, "
                      f"test_f1={test_metrics['f1']:.4f}")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
    
    return results, hyperparams


def run_embedding_sequencer(
    masking_model: str,
    days_given: int,
    days_predicted: int,
    prediction_model: str,
    output_folder: str,
    min_days_per_subject: int = 30
):
    """
    Run the embedding sequencer pipeline.
    
    Args:
        masking_model: The masking model to use (e.g., "masking_10", "masking_30", "masking_50")
        days_given: Number of days of input data (e.g., 3, 5, 7)
        days_predicted: Number of days to predict (e.g., 1, 5, 7, 14)
        prediction_model: Model architecture ("cnn" or "nn")
        output_folder: Directory to save results
        min_days_per_subject: Minimum days of data required per subject (default: 30)
    
    Returns:
        None. Saves results.json to output_folder upon completion.
    """
    # Get timezone and start time
    pst = pytz.timezone('America/Los_Angeles')
    started_at = datetime.now(pst).strftime('%Y-%m-%d %H:%M:%S %Z')
    
    print(f"\n{'='*60}")
    print("EMBEDDING SEQUENCER")
    print(f"{'='*60}")
    print(f"Masking model: {masking_model}")
    print(f"Days given: {days_given}")
    print(f"Days predicted: {days_predicted}")
    print(f"Prediction model: {prediction_model}")
    print(f"Output types: {OUTPUT_TYPES}")
    print(f"Min days per subject: {min_days_per_subject}")
    print(f"Output folder: {output_folder}")
    print(f"Started at: {started_at}")
    print(f"{'='*60}\n")
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load data
    splits, stats, subject_ids = load_all_data(
        masking_model=masking_model,
        days_given=days_given,
        days_predicted=days_predicted,
        min_days_per_subject=min_days_per_subject
    )
    
    print(f"\nTraining {prediction_model.upper()} models for {len(subject_ids)} subjects...")
    print(f"Training config: {TRAINING_CONFIG}")
    
    # Initialize results storage for all metrics
    # Regression tasks: train_loss, test_loss
    # Binary tasks: train_loss, test_loss, and classification metrics
    subject_results = {}
    
    # Regression outputs
    for output_type in ['bpm', 'steps']:
        subject_results[f'train_{output_type}_loss'] = []
        subject_results[f'test_{output_type}_loss'] = []
    
    # Binary classification outputs - store all metrics
    for output_type in ['anxiety', 'stress']:
        subject_results[f'train_{output_type}_loss'] = []
        subject_results[f'test_{output_type}_loss'] = []
        subject_results[f'test_{output_type}_balanced_accuracy'] = []
        subject_results[f'test_{output_type}_accuracy'] = []
        subject_results[f'test_{output_type}_f1'] = []
        subject_results[f'test_{output_type}_precision'] = []
        subject_results[f'test_{output_type}_recall'] = []
        subject_results[f'test_{output_type}_auc'] = []
        subject_results[f'test_{output_type}_n_valid'] = []
    
    subject_index = []
    model_hyperparams = None
    
    # Train for each subject
    for i, user_id in enumerate(tqdm(subject_ids, desc="Training subjects")):
        results, hyperparams = train_and_evaluate_subject(
            user_id=user_id,
            data=splits[user_id],
            model_type=prediction_model,
            days_given=days_given,
            days_predicted=days_predicted,
            stats=stats,
            device=device,
            verbose=False
        )
        
        if model_hyperparams is None:
            model_hyperparams = hyperparams
        
        # Store results for all output types
        for key in subject_results.keys():
            if key in results:
                subject_results[key].append(results[key])
        
        subject_index.append(user_id)
    
    # Compute averages (using nanmean for metrics that may have NaN)
    average_results = {}
    for key, values in subject_results.items():
        if len(values) > 0:
            average_results[key] = float(np.nanmean(values))
        else:
            average_results[key] = float('nan')
    
    # End time
    ended_at = datetime.now(pst).strftime('%Y-%m-%d %H:%M:%S %Z')
    
    # Build output dictionary
    output = {
        'parameters': {
            'masking_model': masking_model,
            'days_given': days_given,
            'days_predicted': days_predicted,
            'prediction_model': prediction_model,
            'min_days_per_subject': min_days_per_subject,
            'output_types': OUTPUT_TYPES
        },
        'started_at': started_at,
        'ended_at': ended_at,
        'average_results': average_results,
        'metadata': {
            'bpm_loss_type': 'mae',
            'steps_loss_type': 'mae',
            'anxiety_loss_type': 'binary_cross_entropy',
            'stress_loss_type': 'binary_cross_entropy',
            'anxiety_metrics': ['balanced_accuracy', 'accuracy', 'f1', 'precision', 'recall', 'auc'],
            'stress_metrics': ['balanced_accuracy', 'accuracy', 'f1', 'precision', 'recall', 'auc'],
            'num_subjects': len(subject_ids),
            'device': str(device)
        },
        'model_params': model_hyperparams,
        'training_config': TRAINING_CONFIG,
        'global_stats': stats,
        'subject_results': subject_results,
        'subject_index': subject_index
    }
    
    # Save results
    os.makedirs(output_folder, exist_ok=True)
    results_path = os.path.join(output_folder, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=4)
    
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print("\n--- Regression Tasks ---")
    print(f"BPM - Test MAE: {average_results['test_bpm_loss']:.4f}")
    print(f"Steps - Test MAE: {average_results['test_steps_loss']:.4f}")
    print("\n--- Binary Classification Tasks ---")
    print(f"Anxiety - Balanced Acc: {average_results.get('test_anxiety_balanced_accuracy', float('nan')):.4f}, "
          f"F1: {average_results.get('test_anxiety_f1', float('nan')):.4f}, "
          f"AUC: {average_results.get('test_anxiety_auc', float('nan')):.4f}")
    print(f"Stress - Balanced Acc: {average_results.get('test_stress_balanced_accuracy', float('nan')):.4f}, "
          f"F1: {average_results.get('test_stress_f1', float('nan')):.4f}, "
          f"AUC: {average_results.get('test_stress_auc', float('nan')):.4f}")
    print(f"\nResults saved to: {results_path}")
    print(f"Ended at: {ended_at}")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Test run with fixed parameters
    import argparse
    
    parser = argparse.ArgumentParser(description="Run embedding sequencer")
    parser.add_argument('--masking-model', default='masking_10', help='Masking model to use')
    parser.add_argument('--days-given', type=int, default=3, help='Number of input days')
    parser.add_argument('--days-predicted', type=int, default=7, help='Number of days to predict')
    parser.add_argument('--prediction-model', default='nn', choices=['cnn', 'nn'], help='Model type')
    parser.add_argument('--output-folder', default='./test_results', help='Output folder')
    parser.add_argument('--min-days', type=int, default=30, help='Minimum days per subject')
    
    args = parser.parse_args()
    
    print(f"Output types being trained: {OUTPUT_TYPES}")
    print("  - bpm: regression (MAE)")
    print("  - steps: regression (MAE)")
    print("  - anxiety: binary classification (BCE)")
    print("  - stress: binary classification (BCE)")
    
    run_embedding_sequencer(
        masking_model=args.masking_model,
        days_given=args.days_given,
        days_predicted=args.days_predicted,
        prediction_model=args.prediction_model,
        output_folder=args.output_folder,
        min_days_per_subject=args.min_days
    )

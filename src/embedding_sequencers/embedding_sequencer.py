"""
Embedding Sequencer

Trains sequence prediction models on pre-computed embeddings.
Per-subject models predicting future biosignals from embedding sequences.
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
    from models import create_model, get_loss_function
else:
    from .data_loader import load_all_data
    from .models import create_model, get_loss_function

import pytz
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


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
        train_Y: np.ndarray (num_samples, days_predicted, 2)
        output_type: 'bpm', 'steps', or 'both'
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
    
    # Prepare targets based on output type
    if output_type == 'bpm':
        targets = train_Y[:, :, 0:1]  # (N, days_predicted, 1)
    elif output_type == 'steps':
        targets = train_Y[:, :, 1:2]  # (N, days_predicted, 1)
    else:  # both
        targets = train_Y  # (N, days_predicted, 2)
    
    # Convert to tensors
    X_tensor = torch.tensor(train_X, dtype=torch.float32).to(device)
    Y_tensor = torch.tensor(targets, dtype=torch.float32).to(device)
    
    # Create data loader
    dataset = TensorDataset(X_tensor, Y_tensor)
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    # Loss and optimizer
    loss_fn = get_loss_function(output_type, stats['bpm_weight'], stats['steps_weight'])
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
        
        avg_loss = epoch_loss / num_batches
        
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
        test_Y: np.ndarray (1, days_predicted, 2)
        output_type: 'bpm', 'steps', or 'both'
        stats: global statistics dict
        device: torch device
    
    Returns:
        loss: test loss value
    """
    model.eval()
    
    # Prepare targets
    if output_type == 'bpm':
        targets = test_Y[:, :, 0:1]
    elif output_type == 'steps':
        targets = test_Y[:, :, 1:2]
    else:
        targets = test_Y
    
    X_tensor = torch.tensor(test_X, dtype=torch.float32).to(device)
    Y_tensor = torch.tensor(targets, dtype=torch.float32).to(device)
    
    loss_fn = get_loss_function(output_type, stats['bpm_weight'], stats['steps_weight'])
    
    with torch.no_grad():
        pred = model(X_tensor)
        loss = loss_fn(pred, Y_tensor)
    
    return loss.item()


def train_and_evaluate_subject(user_id, data, model_type, days_given, days_predicted, 
                                stats, device, verbose=False):
    """
    Train and evaluate all three output types for a single subject.
    
    Returns:
        results: dict with train/test losses for bpm, steps, both
        hyperparams: model hyperparameters
    """
    train_X = data['train_X']
    train_Y = data['train_Y']
    test_X = data['test_X']
    test_Y = data['test_Y']
    
    results = {}
    hyperparams = None
    
    for output_type in ['bpm', 'steps', 'both']:
        # Create fresh model
        model = create_model(model_type, days_given, days_predicted, output_type)
        
        if hyperparams is None:
            hyperparams = model.get_hyperparameters()
        
        # Train
        train_loss = train_model(model, train_X, train_Y, output_type, stats, device)
        
        # Evaluate
        test_loss = evaluate_model(model, test_X, test_Y, output_type, stats, device)
        
        results[f'train_{output_type}_loss'] = train_loss
        results[f'test_{output_type}_loss'] = test_loss
        
        if verbose:
            print(f"    {output_type}: train={train_loss:.4f}, test={test_loss:.4f}")
        
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
    
    # Results storage
    subject_results = {
        'train_bpm_loss': [],
        'train_steps_loss': [],
        'train_both_loss': [],
        'test_bpm_loss': [],
        'test_steps_loss': [],
        'test_both_loss': []
    }
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
        
        # Store results
        subject_results['train_bpm_loss'].append(results['train_bpm_loss'])
        subject_results['train_steps_loss'].append(results['train_steps_loss'])
        subject_results['train_both_loss'].append(results['train_both_loss'])
        subject_results['test_bpm_loss'].append(results['test_bpm_loss'])
        subject_results['test_steps_loss'].append(results['test_steps_loss'])
        subject_results['test_both_loss'].append(results['test_both_loss'])
        subject_index.append(user_id)
    
    # Compute averages
    average_results = {
        'train_bpm_loss': float(np.mean(subject_results['train_bpm_loss'])),
        'train_steps_loss': float(np.mean(subject_results['train_steps_loss'])),
        'train_both_loss': float(np.mean(subject_results['train_both_loss'])),
        'test_bpm_loss': float(np.mean(subject_results['test_bpm_loss'])),
        'test_steps_loss': float(np.mean(subject_results['test_steps_loss'])),
        'test_both_loss': float(np.mean(subject_results['test_both_loss']))
    }
    
    # End time
    ended_at = datetime.now(pst).strftime('%Y-%m-%d %H:%M:%S %Z')
    
    # Build output dictionary
    output = {
        'parameters': {
            'masking_model': masking_model,
            'days_given': days_given,
            'days_predicted': days_predicted,
            'prediction_model': prediction_model,
            'min_days_per_subject': min_days_per_subject
        },
        'started_at': started_at,
        'ended_at': ended_at,
        'average_results': average_results,
        'metadata': {
            'test_bpm_loss_type': 'mse',
            'test_steps_loss_type': 'mae',
            'test_both_loss_type': f'weighted_mse_mae (bpm_weight={stats["bpm_weight"]}, steps_weight={stats["steps_weight"]})',
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
    print(f"Average test BPM loss (MSE): {average_results['test_bpm_loss']:.4f}")
    print(f"Average test Steps loss (MAE): {average_results['test_steps_loss']:.4f}")
    print(f"Average test Both loss (weighted): {average_results['test_both_loss']:.4f}")
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
    parser.add_argument('--prediction-model', default='cnn', choices=['cnn', 'nn'], help='Model type')
    parser.add_argument('--output-folder', default='./test_results', help='Output folder')
    parser.add_argument('--min-days', type=int, default=30, help='Minimum days per subject')
    
    args = parser.parse_args()
    
    run_embedding_sequencer(
        masking_model=args.masking_model,
        days_given=args.days_given,
        days_predicted=args.days_predicted,
        prediction_model=args.prediction_model,
        output_folder=args.output_folder,
        min_days_per_subject=args.min_days
    )

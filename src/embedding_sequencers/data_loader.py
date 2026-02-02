"""
Data loading utilities for embedding sequencers.

Loads pre-computed embeddings and raw biosignal data, computes daily targets.
Supports both regression (bpm, steps) and binary classification (anxiety, stress).
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Add paths for tiles_dataloader
USER_ROOT = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, os.path.join(USER_ROOT, "ssl-physio", "src", "dataloaders"))
physio_data_path = os.path.join(USER_ROOT, "physio-data", "src")
sys.path.insert(0, physio_data_path)

# Override constants before importing tiles_dataloader
import constants
# constants.TILES_HOLDOUT_FITBIT_BASE_FOLDER = "/data1/mjma/tiles-holdout/fitbit"
# constants.TILES_HOLDOUT_LABELS_DEMOG = "/data1/mjma/tiles-holdout/labels/demographics_1.csv"
# constants.TILES_HOLDOUT_LABELS_ANXIETY = "/data1/mjma/tiles-holdout/labels/anxiety.csv"
# constants.TILES_HOLDOUT_LABELS_SHIFT = "/data1/mjma/tiles-holdout/labels/shift.csv"
# constants.TILES_HOLDOUT_LABELS_STRESSD = "/data1/mjma/tiles-holdout/labels/stressd.csv"

from tiles_dataloader import load_tiles_holdout, generate_binary_labels, generate_continuous_labels_day

# Base path for embeddings
EMBEDDINGS_BASE_DIR = "/data1/emilyzho/tiles-2018-processed/tiles-holdout/embeddings"


def load_embeddings_and_index(masking_model: str):
    """
    Load embeddings and index for a given masking model.
    
    Args:
        masking_model: e.g., "masking_10", "masking_30", "masking_50"
    
    Returns:
        embeddings: np.ndarray of shape (N, 128)
        index: list of dicts with keys: key, row, user, date, user_day
    """
    emb_dir = os.path.join(EMBEDDINGS_BASE_DIR, masking_model)
    
    embeddings = np.load(os.path.join(emb_dir, "embeddings.npy"))
    with open(os.path.join(emb_dir, "index.json"), "r") as f:
        index = json.load(f)
    
    return embeddings, index


def load_raw_data_and_compute_targets():
    """
    Load raw biosignal data and compute daily targets (avg_bpm, total_steps).
    Also loads binary labels (anxiety, stress) for each day.
    
    Returns:
        targets_by_user: dict mapping user_id -> dict mapping date -> {
            'bpm': avg_bpm, 'steps': total_steps, 'anxiety': binary, 'stress': binary
        }
        subject_ids: list of subject IDs
        dates: list of dates (needed for binary label generation)
    """
    # Load raw data without scaling to get actual values
    signal_columns = ["bpm", "StepCount"]
    subject_ids, dates, data = load_tiles_holdout(
        signal_columns=signal_columns,
        scale=None,  # No scaling - we want raw values
        window_size=0,  # No smoothing
        debug=False
    )
    
    # Generate binary labels for anxiety and stress
    bpm_values = generate_continuous_labels_day(subject_ids, dates, version="holdout", label_type=constants.Labels.HR)
    steps_values = generate_continuous_labels_day(subject_ids, dates, version="holdout", label_type=constants.Labels.STEPS)
    # rhr_values = generate_continuous_labels_day(subject_ids, dates, version="holdout", label_type=constants.Labels.RHR)
    # sleep_values = generate_continuous_labels_day(subject_ids, dates, version="holdout", label_type=constants.Labels.SLEEP_MINS)
    anxiety_labels = generate_binary_labels(subject_ids, dates, version="holdout", label_type="anxiety")
    stress_labels = generate_binary_labels(subject_ids, dates, version="holdout", label_type="stress")
    
    targets_by_user = defaultdict(dict)
    
    for i, (subject_id, date, day_data) in enumerate(zip(subject_ids, dates, data)):
        bpm_value = bpm_values[i]
        steps_value = steps_values[i]
        
        # Get binary labels (may be NaN if not available)
        anxiety = anxiety_labels[i]
        stress = stress_labels[i]
        
        targets_by_user[subject_id][str(date)] = {
            'bpm': bpm_value,
            'steps': steps_value,
            'anxiety': anxiety,
            'stress': stress
        }
    
    return dict(targets_by_user)


def organize_data_by_subject(embeddings, index, targets_by_user, min_days_per_subject=30):
    """
    Organize embeddings and targets by subject, filtering by minimum days.
    
    Args:
        embeddings: np.ndarray of shape (N, 128)
        index: list of dicts from index.json
        targets_by_user: dict from load_raw_data_and_compute_targets
        min_days_per_subject: minimum number of days required per subject
    
    Returns:
        subject_data: dict mapping user_id -> {
            'embeddings': np.ndarray of shape (num_days, 128) sorted by date,
            'targets': np.ndarray of shape (num_days, 4) [bpm, steps, anxiety, stress],
            'dates': list of date strings sorted chronologically
        }
    """
    # Group index entries by user
    user_entries = defaultdict(list)
    for entry in index:
        user_entries[entry["user"]].append(entry)
    
    subject_data = {}
    
    for user_id, entries in user_entries.items():
        # Skip if not enough days
        if len(entries) < min_days_per_subject:
            continue
        
        # Skip if user not in targets (shouldn't happen but be safe)
        if user_id not in targets_by_user:
            continue
        
        # Sort entries by date
        entries_sorted = sorted(entries, key=lambda x: x["date"])
        
        # Build arrays
        user_embeddings = []
        user_targets = []
        user_dates = []
        
        for entry in entries_sorted:
            date = entry["date"]
            
            # Check if we have targets for this date
            if date not in targets_by_user[user_id]:
                continue
            
            target_dict = targets_by_user[user_id][date]
            user_embeddings.append(embeddings[entry["row"]])
            # Store as [bpm, steps, anxiety, stress]
            user_targets.append([
                target_dict['bpm'],
                target_dict['steps'],
                target_dict['anxiety'],
                target_dict['stress']
            ])
            user_dates.append(date)
        
        # Final check for minimum days after filtering
        if len(user_embeddings) < min_days_per_subject:
            continue
        
        subject_data[user_id] = {
            'embeddings': np.array(user_embeddings),  # (num_days, 128)
            'targets': np.array(user_targets),  # (num_days, 4) [bpm, steps, anxiety, stress]
            'dates': user_dates
        }
    
    return subject_data


def create_sequences(embeddings, targets, days_given, days_predicted):
    """
    Create input/output sequences with sliding window (stride 1).
    
    Args:
        embeddings: np.ndarray of shape (num_days, 128)
        targets: np.ndarray of shape (num_days, 4) [bpm, steps, anxiety, stress]
        days_given: number of input days
        days_predicted: number of output days
    
    Returns:
        X: np.ndarray of shape (num_samples, days_given, 128)
        Y: np.ndarray of shape (num_samples, days_predicted, 4)
    """
    num_days = len(embeddings)
    window_size = days_given + days_predicted
    
    if num_days < window_size:
        return np.array([]), np.array([])
    
    X = []
    Y = []
    
    # Sliding window with stride 1
    for i in range(num_days - window_size + 1):
        X.append(embeddings[i:i + days_given])
        Y.append(targets[i + days_given:i + window_size])
    
    return np.array(X), np.array(Y)


def prepare_train_test_split(subject_data, days_given, days_predicted):
    """
    Prepare train/test splits for each subject.
    
    Test split: last days_predicted days
    Train split: first (total - days_predicted) days
    
    Args:
        subject_data: dict from organize_data_by_subject
        days_given: number of input days
        days_predicted: number of output days
    
    Returns:
        splits: dict mapping user_id -> {
            'train_X': np.ndarray, 'train_Y': np.ndarray,
            'test_X': np.ndarray, 'test_Y': np.ndarray
        }
    """
    splits = {}
    
    for user_id, data in subject_data.items():
        num_days = len(data['embeddings'])
        
        # Split point: everything except last days_predicted for training
        train_end = num_days - days_predicted
        
        # Need at least days_given days in train to create one sample
        if train_end < days_given:
            continue
        
        # Train data
        train_embeddings = data['embeddings'][:train_end]
        train_targets = data['targets'][:train_end]
        
        # Test data: use last days_given + days_predicted days
        # Input: last days_given days of train portion
        # Output: last days_predicted days (the test portion)
        test_X = data['embeddings'][train_end - days_given:train_end]  # (days_given, 128)
        test_Y = data['targets'][train_end:train_end + days_predicted]  # (days_predicted, 4)
        
        # Create training sequences with sliding window
        train_X, train_Y = create_sequences(train_embeddings, train_targets, days_given, days_predicted)
        
        if len(train_X) == 0:
            continue
        
        splits[user_id] = {
            'train_X': train_X,  # (num_train_samples, days_given, 128)
            'train_Y': train_Y,  # (num_train_samples, days_predicted, 4)
            'test_X': test_X[np.newaxis, :, :],  # (1, days_given, 128)
            'test_Y': test_Y[np.newaxis, :, :]   # (1, days_predicted, 4)
        }
    
    return splits


def compute_global_statistics(splits):
    """
    Compute global statistics for loss weighting.
    
    Targets are stored as: [bpm, steps, anxiety, stress]
    - bpm (index 0): regression, use MSE
    - steps (index 1): regression, use MAE
    - anxiety (index 2): binary classification
    - stress (index 3): binary classification
    
    Returns:
        stats: dict with variance/mean stats for regression targets,
               and class balance info for binary targets
    """
    all_bpm = []
    all_steps = []
    all_anxiety = []
    all_stress = []
    
    for user_id, data in splits.items():
        # Use training targets - shape (num_samples, days_predicted, 4)
        all_bpm.extend(data['train_Y'][:, :, 0].flatten())
        all_steps.extend(data['train_Y'][:, :, 1].flatten())
        all_anxiety.extend(data['train_Y'][:, :, 2].flatten())
        all_stress.extend(data['train_Y'][:, :, 3].flatten())
    
    all_bpm = np.array(all_bpm)
    all_steps = np.array(all_steps)
    all_anxiety = np.array(all_anxiety)
    all_stress = np.array(all_stress)
    
    # Filter out NaNs for binary labels
    valid_anxiety = all_anxiety[~np.isnan(all_anxiety)]
    valid_stress = all_stress[~np.isnan(all_stress)]
    
    stats = {
        # Regression stats
        'bpm_var': float(np.var(all_bpm)),
        'bpm_mean': float(np.mean(all_bpm)),
        'bpm_std': float(np.std(all_bpm)),
        'steps_var': float(np.var(all_steps)),
        'steps_mean': float(np.mean(all_steps)),
        'steps_std': float(np.std(all_steps)),
        # Binary classification stats
        'anxiety_positive_rate': float(np.mean(valid_anxiety)) if len(valid_anxiety) > 0 else 0.5,
        'anxiety_num_valid': int(len(valid_anxiety)),
        'stress_positive_rate': float(np.mean(valid_stress)) if len(valid_stress) > 0 else 0.5,
        'stress_num_valid': int(len(valid_stress)),
    }
    
    # Compute weights for regression tasks (kept for backward compatibility)
    raw_bpm_weight = 1.0 / stats['bpm_var']
    raw_steps_weight = 1.0 / stats['steps_std']
    total_weight = raw_bpm_weight + raw_steps_weight
    stats['bpm_weight'] = round(raw_bpm_weight / total_weight, 4)
    stats['steps_weight'] = round(raw_steps_weight / total_weight, 4)
    
    return stats


def load_all_data(masking_model: str, days_given: int, days_predicted: int, min_days_per_subject: int = 30):
    """
    Main entry point: load all data needed for training.
    
    Returns:
        splits: train/test splits per subject
        stats: global statistics for loss weighting
        subject_ids: list of subject IDs that passed filtering
    """
    print(f"Loading embeddings for {masking_model}...")
    embeddings, index = load_embeddings_and_index(masking_model)
    print(f"  Loaded {len(embeddings)} embeddings")
    
    print("Loading raw data, computing daily targets, and loading binary labels...")
    targets_by_user = load_raw_data_and_compute_targets()
    print(f"  Found targets for {len(targets_by_user)} subjects")
    
    print(f"Organizing by subject (min {min_days_per_subject} days)...")
    subject_data = organize_data_by_subject(embeddings, index, targets_by_user, min_days_per_subject)
    print(f"  {len(subject_data)} subjects passed filtering")
    
    print(f"Creating train/test splits (days_given={days_given}, days_predicted={days_predicted})...")
    splits = prepare_train_test_split(subject_data, days_given, days_predicted)
    print(f"  {len(splits)} subjects have valid splits")
    
    print("Computing global statistics...")
    stats = compute_global_statistics(splits)
    print(f"  BPM: mean={stats['bpm_mean']:.2f}, var={stats['bpm_var']:.2f}")
    print(f"  Steps: mean={stats['steps_mean']:.2f}, var={stats['steps_var']:.2f}")
    print(f"  Anxiety: positive_rate={stats['anxiety_positive_rate']:.3f}, n_valid={stats['anxiety_num_valid']}")
    print(f"  Stress: positive_rate={stats['stress_positive_rate']:.3f}, n_valid={stats['stress_num_valid']}")
    
    subject_ids = list(splits.keys())
    
    return splits, stats, subject_ids


if __name__ == "__main__":
    # Test the data loader
    splits, stats, subject_ids = load_all_data(
        masking_model="masking_10",
        days_given=3,
        days_predicted=7,
        min_days_per_subject=30
    )
    
    print(f"\n{'='*60}")
    print("DATA LOADING TEST RESULTS")
    print(f"{'='*60}")
    print(f"Number of subjects: {len(subject_ids)}")
    print(f"First 5 subjects: {subject_ids[:5]}")
    
    # Sample one subject
    sample_user = subject_ids[0]
    sample_data = splits[sample_user]
    print(f"\nSample subject: {sample_user}")
    print(f"  Train X shape: {sample_data['train_X'].shape}")
    print(f"  Train Y shape: {sample_data['train_Y'].shape}  (columns: bpm, steps, anxiety, stress)")
    print(f"  Test X shape: {sample_data['test_X'].shape}")
    print(f"  Test Y shape: {sample_data['test_Y'].shape}")
    
    # Show sample targets
    print(f"\n  Sample train targets (first 3 windows, first day):")
    for i in range(min(3, len(sample_data['train_Y']))):
        y = sample_data['train_Y'][i, 0]  # First predicted day
        print(f"    bpm={y[0]:.1f}, steps={y[1]:.0f}, anxiety={y[2]}, stress={y[3]}")
    
    print(f"\nGlobal stats: {stats}")


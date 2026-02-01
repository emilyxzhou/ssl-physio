"""
Data loading utilities for context_windows MAML experiment.

Loads pre-computed embeddings (S4 or Mamba) and all 5 target labels from labels.npy:
- Binary: stress, anxiety (cols 2-3)
- Regression: sleep, rhr, steps (cols 6-8, already normalized)
"""

import json
import os
import datetime
from collections import defaultdict

import numpy as np

# Base path for tiles-test data
TILES_TEST_DIR = "/data1/mjma/tiles-2018-processed/tiles-test"
EMBEDDINGS_BASE_DIR = os.path.join(TILES_TEST_DIR, "embeddings")

# Target column indices in labels.npy
# Cols: [user_id, date, stress, anxiety, ?, ?, sleep, rhr, steps]
TARGET_NAMES = ['stress', 'anxiety', 'rhr', 'sleep', 'steps']
TARGET_TYPES = {
    'stress': 'binary',
    'anxiety': 'binary', 
    'rhr': 'regression',
    'sleep': 'regression',
    'steps': 'regression'
}


def load_embeddings_and_index(embedding_model: str, masking_ratio: str):
    """
    Load embeddings and index for a given embedding model and masking ratio.
    
    Args:
        embedding_model: "s4" or "mamba"
        masking_ratio: e.g., "masking_10", "masking_30", "masking_50", "masking_70"
    
    Returns:
        embeddings: np.ndarray of shape (N, 128)
        index: list of dicts with keys: key, row, user, date, user_day
    """
    emb_dir = os.path.join(EMBEDDINGS_BASE_DIR, embedding_model, masking_ratio)
    
    embeddings = np.load(os.path.join(emb_dir, "embeddings.npy"))
    with open(os.path.join(emb_dir, "index.json"), "r") as f:
        index = json.load(f)
    
    return embeddings, index


def load_labels():
    """
    Load pre-computed labels from labels.npy.
    
    Returns:
        labels_dict: dict mapping (user_id, date_str) -> {stress, anxiety, rhr, sleep, steps}
    """
    labels_path = os.path.join(TILES_TEST_DIR, "labels.npy")
    labels_arr = np.load(labels_path, allow_pickle=True)
    
    # Build lookup dict: (user_id, date_str) -> all 5 targets
    # Columns: [user_id, date, stress, anxiety, ?, ?, sleep, rhr, steps]
    labels_dict = {}
    for row in labels_arr:
        user_id = row[0]
        date_str = str(row[1])  # Convert datetime.date to string
        
        labels_dict[(user_id, date_str)] = {
            'stress': float(row[2]),
            'anxiety': float(row[3]),
            'sleep': float(row[6]),
            'rhr': float(row[7]),
            'steps': float(row[8])
        }
    
    return labels_dict


def organize_data_by_subject(embeddings, index, labels_dict, min_days_per_subject=30):
    """
    Organize embeddings and all 5 targets by subject.
    
    Args:
        embeddings: np.ndarray of shape (N, 128)
        index: list of dicts from index.json
        labels_dict: dict from load_labels
        min_days_per_subject: minimum days required
    
    Returns:
        subject_data: dict mapping user_id -> {
            'embeddings': np.ndarray (num_days, 128),
            'targets': np.ndarray (num_days, 5) [stress, anxiety, rhr, sleep, steps],
            'dates': list of date strings
        }
    """
    # Group index entries by user
    user_entries = defaultdict(list)
    for entry in index:
        user_entries[entry["user"]].append(entry)
    
    subject_data = {}
    
    for user_id, entries in user_entries.items():
        if len(entries) < min_days_per_subject:
            continue
        
        # Sort by date
        entries_sorted = sorted(entries, key=lambda x: x["date"])
        
        user_embeddings = []
        user_targets = []
        user_dates = []
        
        for entry in entries_sorted:
            date_str = entry["date"]
            
            # Get labels for this user/date
            label = labels_dict.get((user_id, date_str))
            
            if label is None:
                # Skip entries without labels
                continue
            
            user_embeddings.append(embeddings[entry["row"]])
            user_targets.append([
                label['stress'],
                label['anxiety'],
                label['rhr'],
                label['sleep'],
                label['steps']
            ])
            user_dates.append(date_str)
        
        if len(user_embeddings) < min_days_per_subject:
            continue
        
        subject_data[user_id] = {
            'embeddings': np.array(user_embeddings, dtype=np.float32),
            'targets': np.array(user_targets, dtype=np.float32),
            'dates': user_dates
        }
    
    return subject_data


def sample_support_windows(subject_data, target_subject, input_days, output_days, 
                           num_support_subjects=32, samples_per_subject=5, rng=None):
    """
    Sample support set windows from subjects OTHER than the target.
    
    Args:
        subject_data: dict from organize_data_by_subject
        target_subject: user_id to exclude
        input_days: number of input days (N)
        output_days: number of output days to predict
        num_support_subjects: number of other subjects to sample from
        samples_per_subject: number of windows per subject
        rng: numpy random generator for reproducibility
    
    Returns:
        X_support: np.ndarray (num_samples, input_days, 128)
        Y_support: np.ndarray (num_samples, output_days, 5)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Get all subjects except target
    other_subjects = [uid for uid in subject_data.keys() if uid != target_subject]
    
    # Sample support subjects
    if len(other_subjects) < num_support_subjects:
        support_subjects = other_subjects
    else:
        support_subjects = rng.choice(other_subjects, size=num_support_subjects, replace=False)
    
    window_size = input_days + output_days
    X_list = []
    Y_list = []
    
    for subj_id in support_subjects:
        subj_emb = subject_data[subj_id]['embeddings']
        subj_tgt = subject_data[subj_id]['targets']
        num_days = len(subj_emb)
        
        if num_days < window_size:
            continue
        
        # Get valid start positions
        all_starts = list(range(num_days - window_size + 1))
        
        # Try to get non-overlapping windows first
        step = window_size
        non_overlap_starts = list(range(0, num_days - window_size + 1, step))
        
        if len(non_overlap_starts) >= samples_per_subject:
            sampled_starts = rng.choice(non_overlap_starts, size=samples_per_subject, replace=False)
        else:
            # Need overlap, sample from all valid starts
            sampled_starts = rng.choice(all_starts, size=min(samples_per_subject, len(all_starts)), replace=False)
        
        for start in sampled_starts:
            X_list.append(subj_emb[start:start + input_days])
            Y_list.append(subj_tgt[start + input_days:start + window_size])
    
    if not X_list:
        return np.array([]), np.array([])
    
    return np.array(X_list, dtype=np.float32), np.array(Y_list, dtype=np.float32)


def create_query_windows(subject_data, target_subject, input_days, output_days, stride=3):
    """
    Create query set windows for target subject with fixed stride.
    
    Args:
        subject_data: dict from organize_data_by_subject
        target_subject: user_id for query set
        input_days: number of input days
        output_days: number of output days to predict
        stride: step between consecutive windows
    
    Returns:
        X_query: np.ndarray (num_windows, input_days, 128)
        Y_query: np.ndarray (num_windows, output_days, 5)
    """
    subj_emb = subject_data[target_subject]['embeddings']
    subj_tgt = subject_data[target_subject]['targets']
    num_days = len(subj_emb)
    
    window_size = input_days + output_days
    
    if num_days < window_size:
        return np.array([]), np.array([])
    
    X_list = []
    Y_list = []
    
    for start in range(0, num_days - window_size + 1, stride):
        X_list.append(subj_emb[start:start + input_days])
        Y_list.append(subj_tgt[start + input_days:start + window_size])
    
    if not X_list:
        return np.array([]), np.array([])
    
    return np.array(X_list, dtype=np.float32), np.array(Y_list, dtype=np.float32)


def compute_normalization_stats(subject_data):
    """
    Compute normalization statistics for all targets (for logging purposes).
    Note: Regression targets are already normalized in labels.npy.
    
    Returns:
        stats: dict with mean/std for each target
    """
    all_targets = []
    for uid, data in subject_data.items():
        all_targets.append(data['targets'])
    
    all_targets = np.concatenate(all_targets, axis=0)  # (total_days, 5)
    
    stats = {}
    for i, name in enumerate(TARGET_NAMES):
        values = all_targets[:, i]
        valid_values = values[~np.isnan(values)]
        if len(valid_values) > 0:
            stats[f'{name}_mean'] = float(np.mean(valid_values))
            stats[f'{name}_std'] = float(np.std(valid_values)) if np.std(valid_values) > 0 else 1.0
        else:
            stats[f'{name}_mean'] = 0.0
            stats[f'{name}_std'] = 1.0
    
    return stats


def load_all_data(embedding_model: str, masking_ratio: str, min_days_per_subject: int = 30):
    """
    Main entry point: load all data for a given embedding model and masking ratio.
    
    Args:
        embedding_model: "s4" or "mamba"
        masking_ratio: "masking_10", "masking_30", "masking_50", "masking_70"
        min_days_per_subject: minimum days per subject (default 30)
    
    Returns:
        subject_data: organized data by subject
        norm_stats: normalization statistics
        subject_ids: list of valid subject IDs
    """
    print(f"Loading embeddings for {embedding_model}/{masking_ratio}...")
    embeddings, index = load_embeddings_and_index(embedding_model, masking_ratio)
    print(f"  Loaded {len(embeddings)} embeddings")
    
    print("Loading labels from labels.npy...")
    labels_dict = load_labels()
    print(f"  Loaded {len(labels_dict)} label entries")
    
    print(f"Organizing by subject (min {min_days_per_subject} days)...")
    subject_data = organize_data_by_subject(
        embeddings, index, labels_dict, min_days_per_subject
    )
    print(f"  {len(subject_data)} subjects passed filtering")
    
    print("Computing normalization statistics...")
    norm_stats = compute_normalization_stats(subject_data)
    
    subject_ids = list(subject_data.keys())
    
    return subject_data, norm_stats, subject_ids


if __name__ == "__main__":
    # Test data loading
    subject_data, norm_stats, subject_ids = load_all_data(
        embedding_model="s4",
        masking_ratio="masking_10",
        min_days_per_subject=30
    )
    
    print(f"\n{'='*60}")
    print("DATA LOADING TEST")
    print(f"{'='*60}")
    print(f"Number of subjects: {len(subject_ids)}")
    print(f"Normalization stats: {norm_stats}")
    
    # Show sample target coverage
    if subject_ids:
        sample_subj = subject_ids[0]
        targets = subject_data[sample_subj]['targets']
        print(f"\nSample subject {sample_subj[:12]}...:")
        print(f"  Days: {len(targets)}")
        for i, name in enumerate(TARGET_NAMES):
            non_nan = np.sum(~np.isnan(targets[:, i]))
            print(f"  {name}: {non_nan}/{len(targets)} non-NaN")
    
    # Test support/query sampling
    if subject_ids:
        target = subject_ids[0]
        rng = np.random.default_rng(seed=42)
        
        X_sup, Y_sup = sample_support_windows(
            subject_data, target, input_days=3, output_days=5,
            num_support_subjects=32, samples_per_subject=5, rng=rng
        )
        print(f"\nSupport set for target={target[:8]}...:")
        print(f"  X_support shape: {X_sup.shape}")
        print(f"  Y_support shape: {Y_sup.shape}")
        
        X_qry, Y_qry = create_query_windows(
            subject_data, target, input_days=3, output_days=5, stride=3
        )
        print(f"\nQuery set:")
        print(f"  X_query shape: {X_qry.shape}")
        print(f"  Y_query shape: {Y_qry.shape}")

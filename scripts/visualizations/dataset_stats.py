import os
import sys
from pathlib import Path
USER_ROOT = str(Path(__file__).resolve().parents[3])
paths = [
    os.path.join(
        USER_ROOT, "ssl-physio", "src"
    ),
    os.path.join(
        USER_ROOT, "ssl-physio", "src", "dataloaders"
    ),
    os.path.join(
        USER_ROOT, "ssl-physio", "src", "s4_models"
    ),
    os.path.join(
        USER_ROOT, "ssl-physio", "src", "trainers"
    )
]
for path in paths:
    sys.path.insert(0, path)
physio_data_path = os.path.join(
    USER_ROOT, "physio-data", "src"
)
sys.path.append(physio_data_path)

import copy
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import constants

from collections import Counter
from pathlib import Path
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

from tiles_dataloader import TilesDataset, get_data_from_splits, generate_binary_labels, generate_continuous_labels_day


if __name__ == "__main__":
    debug = False
    binary_labels = ["age", "shift", "anxiety", "stress"]

    output_str = ""

    # Demographics ---------------------------------------------------------------------------------------
    pre_subject_ids, pre_dates, pre_data = get_data_from_splits("pretrain")
    subject_ids, dates, data = get_data_from_splits("test")
    total_subjects = pre_subject_ids + subject_ids
    output_str += f"# unique subjects: {len(set(total_subjects))}"

    # Pre-training data ---------------------------------------------------------------------------------------
    num_unique = len(set(pre_subject_ids))
    output_str += "="*70 + "\n"
    output_str += "PRE-TRAINING" + "\n"
    output_str += "="*70 + "\n"
    output_str += f"Number of subjects: {num_unique}.\n"
    output_str += f"Number of samples: {len(pre_subject_ids)}.\n"

    # Pretraining labels ----------------------------------------------------------------------------
    pretrain_df = pd.DataFrame({'ID': pre_subject_ids, 'Date': pre_dates})
    for label_type in binary_labels:
        output_str += "-"*70 + "\n"
        output_str += f"{label_type}\n"
        output_str += "-"*70 + "\n"
        labels = generate_binary_labels(pre_subject_ids, pre_dates, label_type=label_type)
        pretrain_df[label_type] = labels
        labels = np.asarray(labels)
        train_counts = Counter(labels)
        train_counts = list(train_counts.items())
        output_str += f"0: {train_counts[0][1]} | 1: {train_counts[1][1]}\n"

    pretrain_df = pretrain_df.drop_duplicates(subset='ID')
    age_labels = pretrain_df['age'].to_numpy()
    age_counts = Counter(age_labels)
    age_counts = list(age_counts.items())
    output_str += f"# subjects < 40: {age_counts[0][1]} | above 40: {age_counts[1][1]}\n"
    shift_labels = pretrain_df['shift'].to_numpy()
    shift_counts = Counter(shift_labels)
    shift_counts = list(shift_counts.items())
    output_str += f"# day shift: {shift_counts[0][1]} | night shift: {shift_counts[1][1]}\n"

    # Test data -----------------------------------------------------------------------------------------------
    num_unique = len(set(subject_ids))
    output_str += "-"*70 + "\n"
    output_str += "EVALUATION" + "\n"
    output_str += "-"*70 + "\n"
    output_str += f"Number of subjects: {num_unique}.\n"
    output_str += f"Number of samples: {len(subject_ids)}.\n"

    # Downstream evaluation labels ----------------------------------------------------------------------------
    test_df = pd.DataFrame({'ID': subject_ids, 'Date': dates})
    for label_type in binary_labels:
        output_str += "-"*70 + "\n"
        output_str += f"{label_type}\n"
        output_str += "-"*70 + "\n"
        labels = generate_binary_labels(subject_ids, dates, label_type=label_type)
        test_df[label_type] = labels
        labels = np.asarray(labels)
        train_counts = Counter(labels)
        train_counts = list(train_counts.items())
        output_str += f"0: {train_counts[0][1]} | 1: {train_counts[1][1]}\n"

    test_df = test_df.drop_duplicates(subset='ID')
    age_labels = test_df['age'].to_numpy()
    age_counts = Counter(age_labels)
    age_counts = list(age_counts.items())
    output_str += f"# subjects < 40: {age_counts[0][1]} | above 40: {age_counts[1][1]}\n"
    shift_labels = test_df['shift'].to_numpy()
    shift_counts = Counter(shift_labels)
    shift_counts = list(shift_counts.items())
    output_str += f"# day shift: {shift_counts[0][1]} | night shift: {shift_counts[1][1]}\n"
    

    label_types = ["NumberSteps", "RestingHeartRate", "SleepMinutesAsleep"]
    all_labels = generate_continuous_labels_day(subject_ids, dates, label_types=label_types)
    for label_type in label_types:
        output_str += "-"*70 + "\n"
        output_str += f"{label_type}\n"
        output_str += "-"*70 + "\n"
        subject_ids_copy = copy.deepcopy(subject_ids)
        labels = all_labels[label_type]
        nan_indices = [i for i in range(len(labels)) if np.isnan(labels[i])]
        nan_indices.sort(reverse=True)
        for i in nan_indices:
            subject_ids_copy.pop(i)
            labels.pop(i)

        output_str += f"Number of samples for {label_type} after filtering: {len(labels)}\n"

    with open("/home/emilyzho/ssl-physio/scripts/visualizations/dataset_stats.txt", "w") as file:
        file.write(output_str)

    print(output_str)
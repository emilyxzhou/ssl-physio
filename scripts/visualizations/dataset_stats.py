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
from utils import get_kfold_loaders


if __name__ == "__main__":
    debug = False

    output_str = ""

    # Pre-training data ---------------------------------------------------------------------------------------
    subject_ids, dates, data = get_data_from_splits("pretrain")
    num_unique = len(set(subject_ids))
    output_str += "="*70 + "\n"
    output_str += "PRE-TRAINING" + "\n"
    output_str += "="*70 + "\n"
    output_str += f"Number of subjects: {num_unique}.\n"
    output_str += f"Number of samples: {len(subject_ids)}.\n"

    # Pre-training set labels ----------------------------------------------------------------------------
    for label_type in ["age", "shift", "anxiety", "stress"]:
        output_str += "-"*70 + "\n"
        output_str += f"{label_type}\n"
        output_str += "-"*70 + "\n"
        labels = generate_binary_labels(subject_ids, dates, label_type=label_type)
        labels = np.asarray(labels)
        train_counts = Counter(labels)
        train_counts = list(train_counts.items())
        output_str += f"0: {train_counts[0][1]} | 1: {train_counts[1][1]}\n"

    # Test data -----------------------------------------------------------------------------------------------
    subject_ids, dates, data = get_data_from_splits("test")
    num_unique = len(set(subject_ids))
    output_str += "-"*70 + "\n"
    output_str += "EVALUATION" + "\n"
    output_str += "-"*70 + "\n"
    output_str += f"Number of subjects: {num_unique}.\n"
    output_str += f"Number of samples: {len(subject_ids)}.\n"

    # Downstream evaluation labels ----------------------------------------------------------------------------
    for label_type in ["age", "shift", "anxiety", "stress"]:
        output_str += "-"*70 + "\n"
        output_str += f"{label_type}\n"
        output_str += "-"*70 + "\n"
        labels = generate_binary_labels(subject_ids, dates, label_type=label_type)
        labels = np.asarray(labels)
        train_counts = Counter(labels)
        train_counts = list(train_counts.items())
        output_str += f"0: {train_counts[0][1]} | 1: {train_counts[1][1]}\n"
    

    label_types = ["NumberSteps", "RestingHeartRate", "SleepMinutesAsleep"]
    all_labels = generate_continuous_labels_day(subject_ids, dates, label_types=label_types)
    for label_type in label_types:
        output_str += "-"*70 + "\n"
        output_str += f"{label_type}\n"
        output_str += "-"*70 + "\n"
        subject_ids_copy = copy.deepcopy(subject_ids)
        data_copy = copy.deepcopy(data)
        labels = all_labels[label_type]
        nan_indices = [i for i in range(len(labels)) if np.isnan(labels[i])]
        nan_indices.sort(reverse=True)
        for i in nan_indices:
            subject_ids_copy.pop(i)
            data_copy.pop(i)
            labels.pop(i)

        output_str += f"Number of samples for {label_type} after filtering: {len(labels)}\n"

    with open("/home/emilyzho/ssl-physio/scripts/visualizations/dataset_stats.txt", "w") as file:
        file.write(output_str)

    print(output_str)
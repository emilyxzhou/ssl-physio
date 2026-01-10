import os
import pandas as pd
import sys
from pathlib import Path
USER_ROOT = str(Path(__file__).resolve().parents[2])
paths = [
    os.path.join(
        USER_ROOT, "ssl-physio", "src", "dataloaders"
    ),
    os.path.join(
        USER_ROOT, "ssl-physio", "src", "s4-models"
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

import argparse
import logging
import math
import numpy as np
import pprint

import constants

from collections import Counter
from pathlib import Path
from scipy.stats import pearsonr, ConstantInputWarning
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, roc_auc_score, \
    mean_squared_error, mean_absolute_error
from torch import optim
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

from trainer import Trainer, split_k_fold
from tiles_dataloader import load_tiles_open, load_tiles_holdout, TilesDataset, generate_binary_labels, generate_continuous_labels_day


if __name__ == "__main__":
    debug = False

    # Open dataset -----------------------------------------------------------------------------------------------
    signal_columns = [
        # "RMSStdDev_ms", "RRPeakCoverage", "SDNN_ms", "RR0", 
        # "sleepId", "level", 
        "bpm", "StepCount"
    ]
    scale = "mean"
    window_size = 15    # minutes

    subject_ids_open, dates_open, data = load_tiles_open(
        signal_columns=signal_columns,
        scale=scale, window_size=window_size, debug=debug
    )

    print(f"Number of training samples: {len(subject_ids_open)}")
    print(f"Number of training subjects: {len(set(subject_ids_open))}")

    # Binary label distribution
    label_types = [
        "age",
        "shift"
    ]
    for label_type in label_types:
        labels = generate_binary_labels(subject_ids_open, dates_open, version="open", label_type=label_type)
        df = pd.DataFrame({"ID": subject_ids_open, label_type: labels})
        df = df.drop_duplicates(subset=["ID"])
        print(f"{label_type}: {df[label_type].value_counts()}")

    label_types = [
        "anxiety", 
        "stress"
    ]
    for label_type in label_types:
        labels = generate_binary_labels(subject_ids_open, dates_open, version="open", label_type=label_type)
        df = pd.DataFrame({"ID": subject_ids_open, label_type: labels})
        print(f"{label_type}: {df[label_type].value_counts()}")

    # # Continuous labels
    # label_types = [
    #     constants.Labels.HR,
    #     constants.Labels.SDNN,
    #     constants.Labels.STEPS
    # ]
    # for label_type in label_types:
    #     labels = generate_continuous_labels_day(subject_ids_open, dates_open, version="open", label_type=label_type)
    #     df = pd.DataFrame({"ID": subject_ids_open, label_type: labels})

# --------------------------------------------------------------------------------------------------------------------
    # Held-out dataset -----------------------------------------------------------------------------------------------
    signal_columns = [
        # "RMSStdDev_ms", "RRPeakCoverage", "SDNN_ms", "RR0", 
        # "sleepId", "level", 
        "bpm", "StepCount"
    ]
    scale = "mean"
    window_size = 15    # minutes

    subject_ids_holdout, dates_holdout, data = load_tiles_holdout(
        signal_columns=signal_columns,
        scale=scale, window_size=window_size, debug=debug
    )

    print(f"Number of evaluation samples: {len(subject_ids_holdout)}")
    print(f"Number of evaluation subjects: {len(set(subject_ids_holdout))}")

    # Binary label distribution
    label_types = [
        "age",
        "shift"
    ]
    for label_type in label_types:
        labels = generate_binary_labels(subject_ids_holdout, dates_holdout, version="holdout", label_type=label_type)
        df = pd.DataFrame({"ID": subject_ids_holdout, label_type: labels})
        df = df.drop_duplicates(subset=["ID"])
        print(f"{label_type}: {df[label_type].value_counts()}")

    label_types = [
        "anxiety", 
        "stress"
    ]
    for label_type in label_types:
        labels = generate_binary_labels(subject_ids_holdout, dates_holdout, version="holdout", label_type=label_type)
        df = pd.DataFrame({"ID": subject_ids_holdout, label_type: labels})
        print(f"{label_type}: {df[label_type].value_counts()}")

    # Continuous labels

# --------------------------------------------------------------------------------------------------------------------
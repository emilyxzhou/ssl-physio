import os
import sys
from pathlib import Path
USER_ROOT = str(Path(__file__).resolve().parents[2])
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
from utils import get_kfold_loaders


if __name__ == "__main__":
    debug = False

    labels_df = {
        "ID": [],
        "Date": [],
        "age": [],
        "shift": [],
        "anxiety": [],
        "stress": [],
        "NumberSteps": [],
        "RestingHeartRate": [],
        "SleepMinutesAsleep": []
    }

    subject_ids, dates, data = get_data_from_splits("test")
    label_types = ['NumberSteps', 'RestingHeartRate', 'SleepMinutesAsleep']
    all_labels = generate_continuous_labels_day(subject_ids, dates, label_types=label_types)
    for i, label_type in enumerate(label_types):
        labels = all_labels[label_type]
        # subject_ids_copy = copy.deepcopy(subject_ids)
        # dates_copy = copy.deepcopy(dates)
        # nan_indices = [i for i in range(len(labels)) if np.isnan(labels[i])]
        # nan_indices.sort(reverse=True)
        # for idx in nan_indices:
        #     subject_ids_copy.pop(idx)
        #     dates_copy.pop(idx)
        #     labels.pop(idx)

        if i == 0:
            labels_df["ID"].extend(subject_ids)
            labels_df["Date"].extend(dates)
        labels_df[label_type].extend(labels)

    for label_type in ["age", "shift", "anxiety", "stress"]:
        labels = generate_binary_labels(subject_ids, dates, label_type=label_type)
        labels_df[label_type].extend(labels)
    
    labels_df = pd.DataFrame(labels_df).dropna(how="any").sort_values(by=["ID", "Date"]).reset_index(drop=True)
    labels_df = labels_df.to_numpy()

    save_path = "/data1/mjma/tiles-2018-processed/tiles-test/labels.npy"
    np.save(save_path, labels_df)
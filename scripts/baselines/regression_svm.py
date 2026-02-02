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
import math
import numpy as np
import pprint
import random
import signal
import time
import wandb
import yaml

import constants

from collections import Counter
from datetime import datetime
from imblearn.over_sampling import RandomOverSampler
from pathlib import Path
from scipy.stats import pearsonr, ConstantInputWarning
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, roc_auc_score, \
    mean_squared_error, mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVC, SVR
from torchinfo import summary
from tqdm import tqdm

from tiles_dataloader import get_data_from_splits, TilesDataset, generate_binary_labels, generate_continuous_labels_day

SSL_ROOT = os.path.join(USER_ROOT, "ssl-physio")


if __name__ == "__main__":
    debug = False

    # Loading data -----------------------------------------------------------------------------------------------
    signal_columns = [
        # "RMSStdDev_ms", "RRPeakCoverage", "SDNN_ms", "RR0", 
        # "sleepId", "level", 
        "bpm", "StepCount"
    ]
    scale = "mean"
    window_size = 15    # minutes
    label_types = [
        constants.Labels.RHR,
        constants.Labels.STEPS,
        constants.Labels.SLEEP_MINS
    ]

    subject_ids, dates, data = get_data_from_splits()
    all_labels = generate_continuous_labels_day(subject_ids, dates, label_types=label_types)
    
    for label_type in label_types:
        logging.info(f"Label: {label_type} " + "-"*60)

        subject_ids_copy = copy.deepcopy(subject_ids)
        data_copy = copy.deepcopy(data)

        results = {
            "splits": {
                "train_size": [],
                "test_size": [],
                "train_labels": {
                    0: [],
                    1: []
                },
                "test_labels": {
                    0: [],
                    1: []
                }
            },
            "train": {
                "MSE": [],
                "MAE": [],
                "R": [],
                "p": []
            },
            "test": {
                "MSE": [],
                "MAE": [],
                "R": [],
                "p": []
            }
        }

        labels = all_labels[label_type]

        nan_indices = [i for i in range(len(labels)) if np.isnan(labels[i])]
        nan_indices.sort(reverse=True)
        for i in nan_indices:
            subject_ids_copy.pop(i)
            data_copy.pop(i)
            labels.pop(i)

        subject_ids_copy = np.asarray(subject_ids_copy)
        data_copy = np.asarray(data_copy)
        labels = np.asarray(labels)

        group_kfold = GroupKFold(n_splits=5, shuffle=True, random_state=42)

        for i, (train_index, test_index) in enumerate(group_kfold.split(data_copy, labels, subject_ids_copy)):
            train_subject_ids = subject_ids_copy[train_index].tolist()
            train_data = []
            for idx in train_index: train_data.append(data_copy[idx])
            train_labels = labels[train_index]
            train_labels = train_labels.tolist()

            test_subject_ids = subject_ids_copy[test_index].tolist()
            test_data = []
            for idx in test_index: test_data.append(data_copy[idx])
            test_labels = labels[test_index]
            test_labels = test_labels.tolist()

            results["splits"]["train_size"].append(len(train_subject_ids))
            results["splits"]["test_size"].append(len(test_subject_ids))
            
            # Take daily average of inputs
            for i in range(len(train_data)):
                train_data[i] = np.average(train_data[i], axis=0)
            for i in range(len(test_data)):
                test_data[i] = np.average(test_data[i], axis=0)

            # Load model ------------------------------------------------------------------------------------------------
            model = SVR(
                kernel="rbf",
                C=0.1,
                epsilon=0.01
            )
            train_labels = np.asarray(train_labels).reshape(-1, 1).ravel()
            test_labels = np.asarray(test_labels).reshape(-1, 1).ravel()

            model.fit(train_data, train_labels)
            preds = model.predict(test_data)

            mse = mean_squared_error(test_labels, preds)
            mae = mean_absolute_error(test_labels, preds)
            pearsonr_result = pearsonr(test_labels, preds)

            r = pearsonr_result[0]
            p = pearsonr_result[1]
            results["test"]["MSE"].append(mse)
            results["test"]["MAE"].append(mae)
            results["test"]["R"].append(r)
            results["test"]["p"].append(p)

        logging.info(f"Average MSE | MAE | R | p: {np.mean(results["test"]["MSE"])} {np.mean(results["test"]["MAE"])} {np.mean(results["test"]["R"])} {np.mean(results["test"]["p"])}")
        RESULTS_FILE = os.path.join(SSL_ROOT, "results", "baselines", "regression", f"svm_{label_type}.json")
        with open(RESULTS_FILE, "w") as json_file:
            json.dump(results, json_file, indent=4)
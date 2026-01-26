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

import json
import logging
import numpy as np

from collections import Counter
from datetime import datetime
from imblearn.over_sampling import RandomOverSampler
from pathlib import Path
from scipy.stats import pearsonr, ConstantInputWarning
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, roc_auc_score, \
    mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.svm import SVC, SVR
from torchinfo import summary
from torch.utils.data import DataLoader
from tqdm import tqdm

from tiles_dataloader import TilesDataset, get_data_from_splits, generate_binary_labels
from utils import stratified_group_split, get_kfold_loaders

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
        # "age", "shift", "anxiety", "stress"
        "stress"
    ]
    num_folds = 5
    for label_type in label_types:
        logging.info(f"Label: {label_type} " + "-"*80)

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
                "ACC": [],
                "bACC": [],
                "F1": [],
                "AUC": []
            },
            "test": {
                "ACC": [],
                "bACC": [],
                "F1": [],
                "AUC": []
            }
        }

        subject_ids, dates, data = get_data_from_splits()
        labels = generate_binary_labels(subject_ids, dates, label_type=label_type)

        subject_ids = np.asarray(subject_ids)
        data = np.asarray(data)
        labels = np.asarray(labels)

        group_kfold = GroupKFold(n_splits=5, shuffle=True, random_state=42)

        for i, (train_index, test_index) in enumerate(group_kfold.split(data, labels, subject_ids)):
            train_arr = labels[train_index]
            train_counts = Counter(train_arr)
            train_counts = list(train_counts.items())
            test_arr = labels[test_index]
            test_counts = Counter(test_arr)
            test_counts = list(test_counts.items())

            train_data = data[train_index]
            test_data = data[test_index]

            train_labels = labels[train_index]
            test_labels = labels[test_index]

            results["splits"]["train_size"].append(train_arr.size)
            results["splits"]["test_size"].append(test_arr.size)
            results["splits"]["train_labels"][0].append(train_counts[0][1])
            results["splits"]["train_labels"][1].append(train_counts[1][1])
            results["splits"]["test_labels"][0].append(test_counts[0][1])
            results["splits"]["test_labels"][1].append(test_counts[1][1])

            # Take daily average of inputs
            train_data = np.mean(train_data, axis=1)
            test_data = np.mean(test_data, axis=1)

            # Load model ------------------------------------------------------------------------------------------------
            model = SVC(probability=True, random_state=42)
            train_labels = np.asarray(train_labels).reshape(-1, 1).ravel()
            test_labels = np.asarray(test_labels).reshape(-1, 1).ravel()

            model.fit(train_data, train_labels)
            probs = model.predict(test_data)
            threshold = 0.5
            preds = (probs >= threshold).astype(int)

            acc = accuracy_score(test_labels, preds)
            bacc = balanced_accuracy_score(test_labels, preds)
            f1 = f1_score(test_labels, preds)
            auc = roc_auc_score(test_labels, probs)

            logging.info(f"ACC | bACC | F1 | AUC: {acc} {bacc} {f1} {auc}")

            logging.info(f"Training labels: {train_counts}")
            logging.info(f"Testing labels: {test_counts}")

            results["test"]["ACC"].append(acc)
            results["test"]["bACC"].append(bacc)
            results["test"]["F1"].append(f1)
            results["test"]["AUC"].append(auc)

        logging.info(f"Average ACC | bACC | F1 | AUC: {np.mean(results["test"]["ACC"])} {np.mean(results["test"]["bACC"])} {np.mean(results["test"]["F1"])} {np.mean(results["test"]["AUC"])}")
        RESULTS_FILE = os.path.join(SSL_ROOT, "results", "baselines", "classification", f"svm_{label_type}.json")
        with open(RESULTS_FILE, "w") as json_file:
            json.dump(results, json_file, indent=4)
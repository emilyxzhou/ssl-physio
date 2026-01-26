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
from sklearn.svm import SVC, SVR
from torchinfo import summary
from tqdm import tqdm

from tiles_dataloader import load_tiles_open, load_tiles_holdout, TilesDataset, generate_binary_labels, generate_continuous_labels_day

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
        constants.Labels.HR,
        constants.Labels.SDNN,
        constants.Labels.RHR,
        constants.Labels.STEPS,
        constants.Labels.SLEEP_MINS
    ]
    num_folds = 5
    for label_type in label_types:
        logging.info(f"Label: {label_type} " + "-"*80)

        subject_ids, dates, data = load_tiles_holdout(
            signal_columns=signal_columns,
            scale=scale, window_size=window_size, debug=debug
        )
        labels = generate_continuous_labels_day(subject_ids, dates, version="holdout", label_type=label_type, debug=debug)
        nan_indices = [i for i in range(len(labels)) if np.isnan(labels[i])]
        nan_indices.sort(reverse=True)
        for i in nan_indices:
            subject_ids.pop(i)
            data.pop(i)
            labels.pop(i)

        # 5 folds, randomly sampling 700 samples as the training set and using the remaining as the test set
        # subject_id_folds, data_folds, labels_folds = split_k_fold(subject_ids, data, labels, num_folds=num_folds, seed=37)
        results = {
                "splits": {
                    "train_size": [],
                    "test_size": [],
                },
                "test": {
                    "MSE": [],
                    "MAE": [],
                    "R": [],
                    "p": []
                }
            }

        for i in range(num_folds):
            logging.info(f"Fold {i+1} " + "-"*80)
            train_subject_ids, train_data, train_labels = list(), list(), list()
            test_subject_ids, test_data, test_labels = list(), list(), list()

            # for j in range(num_folds):
            #     if j != i: 
            #         train_subject_ids.extend(subject_id_folds[j])
            #         train_data.extend(data_folds[j])
            #         train_labels.extend(labels_folds[j])
            #     else: 
            #         test_subject_ids.extend(subject_id_folds[j])
            #         test_data.extend(data_folds[j])
            #         test_labels.extend(labels_folds[j])

            random.seed(42*i)
            num_train_samples = int(len(subject_ids) * 0.09)
            logging.info(f"# training samples: {num_train_samples}")
            logging.info(f"# testing samples: {len(subject_ids) - num_train_samples}")
            train_indices = random.sample(list(range(len(subject_ids))), num_train_samples)
            for idx in range(len(subject_ids)):
                if idx in train_indices:
                    train_subject_ids.append(subject_ids[idx])
                    train_data.append(data[idx])
                    train_labels.append(labels[idx])
                else:
                    test_subject_ids.append(subject_ids[idx])
                    test_data.append(data[idx])
                    test_labels.append(labels[idx])

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
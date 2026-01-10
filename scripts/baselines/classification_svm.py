import os
import sys
from pathlib import Path
USER_ROOT = str(Path(__file__).resolve().parents[3])
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
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR
from torchinfo import summary
from tqdm import tqdm

from trainer import Trainer, split_k_fold
from tiles_dataloader import load_tiles_open, load_tiles_holdout, TilesDataset, generate_binary_labels, generate_continuous_labels_day


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
        "age", "shift", "anxiety", "stress"
    ]
    num_folds = 5
    for label_type in label_types:
        logging.info(f"Label: {label_type} " + "-"*80)

        subject_ids, dates, data = load_tiles_holdout(
            signal_columns=signal_columns,
            scale=scale, window_size=window_size, debug=debug
        )
        labels = generate_binary_labels(subject_ids, dates, version="holdout", label_type=label_type)

        # 5 folds, randomly sampling 700 samples as the training set and using the remaining as the test set
        # subject_id_folds, data_folds, labels_folds = split_k_fold(subject_ids, data, labels, num_folds=num_folds, seed=37)
        accs = list()
        baccs = list()
        f1s = list()
        aucs = list()

        for i in range(num_folds):
            logging.info(f"Fold {i+1} " + "-"*80)
            # train_subject_ids, train_data, train_labels = list(), list(), list()
            # test_subject_ids, test_data, test_labels = list(), list(), list()
            # for j in range(num_folds):
            #     if j != i: 
            #         train_subject_ids.extend(subject_id_folds[j])
            #         train_data.extend(data_folds[j])
            #         train_labels.extend(labels_folds[j])
            #     else: 
            #         test_subject_ids.extend(subject_id_folds[j])
            #         test_data.extend(data_folds[j])
            #         test_labels.extend(labels_folds[j])

            train_subject_ids, test_subject_ids, train_data, test_data, train_labels, test_labels = train_test_split(
                subject_ids, data, labels,
                test_size=0.86,         
                stratify=labels,
                random_state=42*i
            )

            # Balance training labels
            # sampler = RandomOverSampler(sampling_strategy=0.8)
            # temp_data = np.array(list(range(len(train_data)))).reshape(-1, 1)
            # train_labels = np.array(train_labels)
            # temp_data, train_labels = sampler.fit_resample(temp_data, train_labels)
            # temp_data = temp_data.flatten()
            # resampled_data, resampled_subjects = list(), list()
            # for i in temp_data:
            #     resampled_data.append(train_data[i])
            #     resampled_subjects.append(train_subject_ids[i])

            train_arr = np.array(train_labels)
            train_counts = Counter(train_arr)
            test_arr = np.array(test_labels)
            test_counts = Counter(test_arr)
            
            logging.info(f"Training labels: {train_counts}")
            logging.info(f"Testing labels: {test_counts}")

            # Take daily average of inputs
            for i in range(len(train_data)):
                train_data[i] = np.average(train_data[i], axis=0)
            for i in range(len(test_data)):
                test_data[i] = np.average(test_data[i], axis=0)

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

            accs.append(acc)
            baccs.append(bacc)
            f1s.append(f1)
            aucs.append(auc)

        logging.info(f"Average ACC | bACC | F1 | AUC: {np.mean(accs)} {np.mean(baccs)} {np.mean(f1s)} {np.mean(aucs)}")
# End-to-end classification with CNN
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
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml

import constants

from collections import Counter
from datetime import datetime
from imblearn.over_sampling import RandomOverSampler
from pathlib import Path
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

from trainer import Trainer, split_k_fold
from tiles_dataloader import load_tiles_open, load_tiles_holdout, TilesDataset, generate_binary_labels, generate_continuous_labels_day

from s4model import S4Model

# Define logging console
import logging
logging.basicConfig(
    format="%(asctime)s %(levelname)-3s ==> %(message)s", 
    level=logging.INFO, 
    datefmt="%Y-%m-%d %H:%M:%S"
)

os.environ["S4_FAST_CAUCHY"] = "0"
os.environ["S4_FAST_VAND"] = "0"
os.environ["S4_BACKEND"] = "keops"   # or "keops" if you installed pykeops


def train_epoch(
    dataloader, 
    model, 
    device, 
    optimizer,
    epoch
):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_labels = []
    total_preds = []
    total_probs = []
    for batch_idx, batch_data in enumerate(dataloader):
        model.zero_grad()
        optimizer.zero_grad()

        # Transfer to GPU
        batch, subject_ids, labels = batch_data
        batch = batch.to(device)                  # original shape (batch_size, 1440, num_features)
        batch = torch.transpose(batch, 1, 2)    # new shape (batch_size, num_features, 1440)

        preds = model(batch).flatten()
        loss = criterion(preds.float(), labels.float()).to(device)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        threshold = 0.5
        probabilities = torch.sigmoid(preds)
        preds = (probabilities >= threshold).int()
        preds = preds.detach().cpu().numpy().astype(int)
        labels = labels.detach().cpu().numpy().astype(int)
        probabilities = probabilities.detach().cpu().numpy().astype(float)
        total_labels.extend(labels)
        total_preds.extend(preds)
        total_probs.extend(probabilities)

        acc = accuracy_score(total_labels, total_preds)
        bacc = balanced_accuracy_score(total_labels, total_preds)
        f1 = f1_score(total_labels, total_preds, average="macro")
        auc = roc_auc_score(total_labels, total_probs)
        
        # if (epoch+1 % 10 == 0) and (batch_idx+1 % 50 == 0 or batch_idx+1 == len(dataloader)):
        # if (batch_idx+1 % 50 == 0 or batch_idx+1 == len(dataloader)):
        #     logging.info(f"Current ACC at epoch {epoch+1}, step {batch_idx+1}/{len(dataloader)} {acc}")
        #     logging.info(f"Current bACC at epoch {epoch+1}, step {batch_idx+1}/{len(dataloader)} {bacc}")
        #     logging.info(f"Current F1 at epoch {epoch+1}, step {batch_idx+1}/{len(dataloader)} {f1}")
        #     logging.info(f"Current AUC at epoch {epoch+1}, step {batch_idx+1}/{len(dataloader)} {auc}")
        #     logging.info(f"------------------------------------------------------------")

    return acc, bacc, f1, auc


def validate_epoch(
    dataloader, 
    model, 
    device,
    epoch,
    split:  str="test"
):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_labels = []
    total_preds = []
    total_probs = []
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            # Transfer to GPU
            batch, subject_ids, labels = batch_data
            batch = batch.to(device)                  # original shape (batch_size, 1440, num_features)
            batch = torch.transpose(batch, 1, 2)    # new shape (batch_size, num_features, 1440)

            preds = model(batch)
            loss = criterion(preds.flatten().float(), labels.float()).to(device)
            preds = preds.to(device)
            labels = labels.to(device)

            threshold = 0.5
            probabilities = torch.sigmoid(preds)
            preds = (probabilities >= threshold).int()
            preds = preds.detach().cpu().numpy().astype(int)
            labels = labels.detach().cpu().numpy().astype(int)
            probabilities = probabilities.detach().cpu().numpy().astype(float)
            total_labels.extend(labels)
            total_preds.extend(preds)
            total_probs.extend(probabilities)

            acc = accuracy_score(total_labels, total_preds)
            bacc = balanced_accuracy_score(total_labels, total_preds)
            f1 = f1_score(total_labels, total_preds, average="macro")
            auc = roc_auc_score(total_labels, total_probs)

            # if (epoch+1 % 10 == 0) and (batch_idx+1 % 50 == 0 or batch_idx+1 == len(dataloader)):
            if (batch_idx+1 % 50 == 0 or batch_idx+1 == len(dataloader)):
                # logging.info(f"Current ACC at epoch {epoch+1}, step {batch_idx+1}/{len(dataloader)} {acc}")
                # logging.info(f"Current bACC at epoch {epoch+1}, step {batch_idx+1}/{len(dataloader)} {bacc}")
                # logging.info(f"Current F1 at epoch {epoch+1}, step {batch_idx+1}/{len(dataloader)} {f1}")
                # logging.info(f"Current AUC at epoch {epoch+1}, step {batch_idx+1}/{len(dataloader)} {auc}")
                logging.info(f"ACC | bACC | F1 | AUC: {acc} {bacc} {f1} {auc}")
                logging.info(f"------------------------------------------------------------")
    return acc, bacc, f1, auc


if __name__ == "__main__":
    debug = False

    # Find device
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.set_default_device("cuda:0")
        print("Default device set to CUDA.")
    else:
        print("CUDA not available.")
    torch.set_default_dtype(torch.float64)

    # Define parameters ----------------------------------------------------------------------------------------------------
    # Training variables
    epochs = 100
    batch_size = 32

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

            tiles_holdout_train = TilesDataset(
                subject_ids=train_subject_ids, data=train_data, labels=train_labels
            )
            train_dataloader = DataLoader(tiles_holdout_train, batch_size=batch_size, num_workers=0, shuffle=True, generator=torch.Generator(device=device))

            tiles_holdout_test = TilesDataset(
                subject_ids=test_subject_ids, data=test_data, labels=test_labels
            )
            test_dataloader = DataLoader(tiles_holdout_test, batch_size=batch_size, num_workers=0, shuffle=False)

            num_train_samples = len(tiles_holdout_train)
            num_test_samples = len(tiles_holdout_test)


            # Load model ------------------------------------------------------------------------------------------------
            model = S4Model(
                d_input=1440,
                d_output=1,
                d_model=128,
                n_layers=6,
                dropout=0.3,
                prenorm=False,
                pooling=True
            )
            if i == 0: summary(model)

            optimizer = optim.AdamW(
                model.parameters(),
                # lr=5e-3,
                lr=1e-5,
                # weight_decay=1e-4,
                # betas=(0.9, 0.95)
            )

            for epoch in range(epochs):
                train_epoch(
                    train_dataloader, model, device, optimizer, epoch=epoch
                )
                
            acc, bacc, f1, auc = validate_epoch(
                test_dataloader, model, device, epoch=0, 
                split="test"
            )

            logging.info(f"Training labels: {train_counts}")
            logging.info(f"Testing labels: {test_counts}")

            accs.append(acc)
            baccs.append(bacc)
            f1s.append(f1)
            aucs.append(auc)

        logging.info(f"\nAverage ACC: {np.mean(accs)}")
        logging.info(f"\nAverage bACC: {np.mean(baccs)}")
        logging.info(f"\nAverage F1: {np.mean(f1s)}")
        logging.info(f"\nAverage AUC: {np.mean(aucs)}")

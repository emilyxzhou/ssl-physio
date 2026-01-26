# End-to-end classification with CNN
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
        USER_ROOT, "ssl-physio", "src", "mamba"
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
import torch
import torch.nn as nn
import torch.nn.functional as F
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

from mamba_mae import MambaModel
from tiles_dataloader import TilesDataset, get_data_from_splits, generate_binary_labels
from utils import get_kfold_loaders

# Define logging console
import logging
logging.basicConfig(
    format="%(asctime)s %(levelname)-3s ==> %(message)s", 
    level=logging.INFO, 
    datefmt="%Y-%m-%d %H:%M:%S"
)

SSL_ROOT = os.path.join(USER_ROOT, "ssl-physio")


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
            probabilities = probabilities.detach().cpu().numpy().astype(float)
            labels = labels.detach().cpu().numpy().astype(int)
            total_labels.extend(labels)
            total_preds.extend(preds)
            total_probs.extend(probabilities)

            acc = accuracy_score(total_labels, total_preds)
            bacc = balanced_accuracy_score(total_labels, total_preds)
            f1 = f1_score(total_labels, total_preds, average="macro")
            auc = roc_auc_score(total_labels, total_probs)

            # if (epoch+1 % 10 == 0) and (batch_idx+1 % 50 == 0 or batch_idx+1 == len(dataloader)):
            if (batch_idx+1 % 50 == 0 or batch_idx+1 == len(dataloader)):
            #     logging.info(f"Current ACC at epoch {epoch+1}, step {batch_idx+1}/{len(dataloader)} {acc}")
            #     logging.info(f"Current bACC at epoch {epoch+1}, step {batch_idx+1}/{len(dataloader)} {bacc}")
            #     logging.info(f"Current F1 at epoch {epoch+1}, step {batch_idx+1}/{len(dataloader)} {f1}")
            #     logging.info(f"Current AUC at epoch {epoch+1}, step {batch_idx+1}/{len(dataloader)} {auc}")
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
    # torch.set_default_dtype(torch.float64)


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
        # "stress"
    ]

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
        tiles_test = TilesDataset(subject_ids, data, labels)
        test_dataloader = DataLoader(tiles_test, batch_size=batch_size, num_workers=0, shuffle=True, generator=torch.Generator(device=device))

        folds = get_kfold_loaders(test_dataloader)

        for fold_idx, (train_dataloader, test_dataloader) in enumerate(folds):
            
            train_arr = np.array(train_dataloader.dataset.dataset.labels)
            train_counts = Counter(train_arr)
            train_counts = list(train_counts.items())
            test_arr = np.array(test_dataloader.dataset.dataset.labels)
            test_counts = Counter(test_arr)
            test_counts = list(test_counts.items())

            results["splits"]["train_size"].append(train_arr.size)
            results["splits"]["test_size"].append(test_arr.size)
            results["splits"]["train_labels"][0].append(train_counts[0][1])
            results["splits"]["train_labels"][1].append(train_counts[1][1])
            results["splits"]["test_labels"][0].append(test_counts[0][1])
            results["splits"]["test_labels"][1].append(test_counts[1][1])

            # Load model ------------------------------------------------------------------------------------------------
            model = MambaModel(
                d_input=1440,
                d_output=1,
                d_model=128,
                n_layers=6,
                pooling=True
            )
            if fold_idx == 0: summary(model)

            optimizer = optim.AdamW(
                model.parameters(),
                # lr=5e-3,
                lr=1e-5,
                # weight_decay=1e-4,
                # betas=(0.9, 0.95)
            )

            for epoch in range(epochs):
                acc, bacc, f1, auc = train_epoch(
                    train_dataloader, model, device, optimizer, epoch=epoch
                )
                results["train"]["ACC"].append(acc)
                results["train"]["bACC"].append(bacc)
                results["train"]["F1"].append(f1)
                results["train"]["AUC"].append(auc)
                
            acc, bacc, f1, auc = validate_epoch(
                test_dataloader, model, device, epoch=0
            )

            results["test"]["ACC"].append(acc)
            results["test"]["bACC"].append(bacc)
            results["test"]["F1"].append(f1)
            results["test"]["AUC"].append(auc)

        logging.info(f"Average ACC | bACC | F1 | AUC: {np.mean(results["test"]["ACC"])} {np.mean(results["test"]["bACC"])} {np.mean(results["test"]["F1"])} {np.mean(results["test"]["AUC"])}")
        RESULTS_FILE = os.path.join(SSL_ROOT, "results", "baselines", "classification", f"mamba_{label_type}.json")
        with open(RESULTS_FILE, "w") as json_file:
            json.dump(results, json_file, indent=4)

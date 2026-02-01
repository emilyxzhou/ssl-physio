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
import argparse
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import Counter
from datetime import datetime
from pathlib import Path
from scipy.stats import pearsonr, ConstantInputWarning
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import optim
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

from tiles_dataloader import get_embeddings_from_file, TilesDataset, generate_binary_labels, generate_continuous_labels_day

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", category=ConstantInputWarning, message="An input array is constant; the correlation coefficient is not defined.")
warnings.filterwarnings(action="ignore", category=UndefinedMetricWarning, message="Only one class is present in y_true. ROC AUC score is not defined in that case.")
warnings.filterwarnings(action="ignore", category=UserWarning, message="A single label was found in 'y_true' and 'y_pred'. For the confusion matrix to have the correct shape, use the 'labels' parameter to pass all known labels.")
warnings.filterwarnings(action="ignore", category=UserWarning, message="y_pred contains classes not in y_true")

# Define logging console
import logging
logging.basicConfig(
    level=logging.INFO, 
    # # datefmt="%Y-%m-%d %H:%M:%S"
)

SSL_ROOT = os.path.join(USER_ROOT, "ssl-physio")
MODELS_BASE_PATH = f"{USER_ROOT}/ssl-physio/models/reconstruction"


class LinearHead(nn.Module):
    def __init__(self, d_input: int, sequence_len=1440):
        super().__init__()
        # Flatten
        dummy_input = torch.randn(1, d_input, sequence_len)
        self._to_linear = dummy_input.shape[1] * dummy_input.shape[2]    # channels * final_sequence_length

        self.fc = nn.Linear(self._to_linear, 1)

    def forward(self, x):
        # x = x.view(-1, self._to_linear)
        x = torch.flatten(x)
        x = self.fc(x)
        return x
    

class CNN(nn.Module):

    def __init__(self, d_input: int, sequence_len=1440, hidden_dim=64):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=d_input, out_channels=128, kernel_size=3, padding="same")
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding="same")
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding="same")
        self.relu3 = nn.ReLU()
        self.pool3 = nn.AvgPool1d(kernel_size=2)

        dummy_input = torch.randn(1, d_input, sequence_len)
        x = self.pool1(self.relu1(self.conv1(dummy_input)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        self._to_linear = x.shape[1] * x.shape[2]    # channels * final_sequence_length
        self.fc = nn.Linear(self._to_linear, 1)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(-1, self._to_linear)
        x = self.fc(x)
        return x


def train_epoch(
    dataloader, 
    model, 
    device, 
    optimizer,
    eval_type="binary"
):
    model.train()
    if eval_type == "binary": criterion = nn.BCEWithLogitsLoss()
    else: criterion = nn.L1Loss()
    total_labels = []
    total_preds = []
    total_probs = []
        
    for batch_idx, batch_data in enumerate(dataloader):
        model.zero_grad()
        optimizer.zero_grad()

        # Transfer to GPU
        batch, subject_ids, labels = batch_data
        batch = batch.to(device)
        # batch = torch.transpose(batch, 1, 2)    # new shape (batch_size, num_features, 1440)                  

        preds = model(batch).flatten()
        loss = criterion(preds.float(), labels.float()).to(device)
        preds = preds.to(device)
        labels = labels.to(device)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if eval_type == "binary":
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

        else:
            preds = preds.detach().cpu().numpy().astype(float)
            labels = labels.detach().cpu().numpy().astype(float)
            total_labels.extend(labels)
            total_preds.extend(preds)

            mse = mean_squared_error(total_labels, total_preds)
            mae = mean_absolute_error(total_labels, total_preds)
            pearsonr_result = pearsonr(total_labels, total_preds)
        
    if eval_type == "binary": return acc, bacc, f1, auc
    else: return mse, mae, pearsonr_result


def validate_epoch(
    dataloader, 
    model, 
    device,
    eval_type="binary"
):
    model.eval()
    if eval_type == "binary": criterion = nn.BCEWithLogitsLoss()
    else: criterion = nn.L1Loss()
    total_labels = []
    total_preds = []
    total_probs = []
        
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            # Transfer to GPU
            batch, subject_ids, labels = batch_data
            batch = batch.to(device)
            # batch = torch.transpose(batch, 1, 2)    # new shape (batch_size, num_features, 1440)         

            preds = model(batch).flatten()
            loss = criterion(preds.flatten().float(), labels.float()).to(device)
            preds = preds.to(device)
            labels = labels.to(device)

            if eval_type == "binary":
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

            else:
                preds = preds.detach().cpu().numpy().astype(float)
                labels = labels.detach().cpu().numpy().astype(float)
                total_labels.extend(labels)
                total_preds.extend(preds)

                mse = mean_squared_error(total_labels, total_preds)
                mae = mean_absolute_error(total_labels, total_preds)
                pearsonr_result = pearsonr(total_labels, total_preds)
        
    if eval_type == "binary": return acc, bacc, f1, auc
    else: return mse, mae, pearsonr_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAE downstream evaluation.")
    parser.add_argument("--device", "-d", default="cuda:1")
    parser.add_argument("--method", "-m", default="raw")
    parser.add_argument("--encoder", "-enc", default="cnn")

    parser.add_argument("--batch_size", "-b", default=32)
    parser.add_argument("--epochs", "-e", default=100)
    parser.add_argument("--classifier", "-c", default="cnn")

    args = parser.parse_args()
    device = args.device
    encoder = args.encoder
    method = args.method
    batch_size = args.batch_size
    epochs = args.epochs
    classifier = args.classifier
    
    # Find device
    device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.set_default_device(device)
        logging.info("\nDefault device set to CUDA.")
    else:
        logging.info("\nCUDA not available, using CPU.")
    torch.set_default_dtype(torch.float)

    binary_labels = ["age", "shift", "anxiety", "stress"]
    continuous_labels = ["NumberSteps", "RestingHeartRate", "SleepMinutesAsleep"]

    model_types = ["s4", "mamba"]
    mask_pcts = [
        10, 
        30, 
        50, 
        70
    ]
    for model_type in model_types:
        for mask_pct in mask_pcts:
            logging.info("="*50)
            logging.info(f"{model_type} | {mask_pct}% masking | Encoder: {encoder}")
            logging.info("="*50)
            subject_ids, dates, data = get_embeddings_from_file(model_type, mask_pct, method=method, encoder=encoder)
            d_input = data[0].shape[0]
            sequence_len = data[0].shape[1]
            
            # Classification ------------------------------------------------------------
            for label_type in binary_labels:
                logging.info("-"*50)
                logging.info(f"Label: {label_type}")
                logging.info("-"*50)
                logging.info(f"Starting at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")

                labels = generate_binary_labels(subject_ids, dates, label_type=label_type)
                subject_ids_arr = np.asarray(subject_ids)
                data_arr = np.asarray(data)
                labels = np.asarray(labels)

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

                group_kfold = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=13)
                idxs = group_kfold.split(data_arr, labels, subject_ids_arr)
                for i, (train_index, test_index) in enumerate(idxs):
                    train_subject_ids = subject_ids_arr[train_index].tolist()
                    train_data = []
                    for idx in train_index: train_data.append(data_arr[idx])
                    train_labels = labels[train_index]
                    train_counts = Counter(train_labels)
                    train_counts = list(train_counts.items())
                    train_labels = train_labels.tolist()

                    test_subject_ids = subject_ids_arr[test_index].tolist()
                    test_data = []
                    for idx in test_index: test_data.append(data_arr[idx])
                    test_labels = labels[test_index]
                    test_counts = Counter(test_labels)
                    test_counts = list(test_counts.items())
                    test_labels = test_labels.tolist()

                    train_dataset = TilesDataset(train_subject_ids, train_data, train_labels)
                    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True, generator=torch.Generator(device=device))
                    test_dataset = TilesDataset(test_subject_ids, test_data, test_labels)
                    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, shuffle=False, generator=torch.Generator(device=device))

                    results["splits"]["train_size"].append(len(train_labels))
                    results["splits"]["test_size"].append(len(test_labels))
                    results["splits"]["train_labels"][0].append(train_counts[0][1])
                    results["splits"]["train_labels"][1].append(train_counts[1][1])
                    results["splits"]["test_labels"][0].append(test_counts[0][1])
                    results["splits"]["test_labels"][1].append(test_counts[1][1])
                    logging.info(f"Training on {len(train_labels)} samples, testing on {len(test_labels)}.")

                    if classifier == "lin": model = LinearHead(d_input=d_input, sequence_len=sequence_len)
                    elif classifier == "cnn": model = CNN(d_input=d_input, sequence_len=sequence_len)

                    optimizer = optim.AdamW(
                        model.parameters(),
                        # lr=5e-3,
                        lr=1e-3,
                        # weight_decay=1e-4,
                        # betas=(0.9, 0.95)
                    )

                    for epoch in range(epochs):
                        acc, bacc, f1, auc = train_epoch(
                            train_dataloader, model, device, optimizer, eval_type="binary"
                        )
                        results["train"]["ACC"].append(acc)
                        results["train"]["bACC"].append(bacc)
                        results["train"]["F1"].append(f1)
                        results["train"]["AUC"].append(auc)
                        
                    acc, bacc, f1, auc = validate_epoch(
                        test_dataloader, model, device, eval_type="binary"
                    )

                    # logging.info(f"Training labels: {train_counts}")
                    # logging.info(f"Testing labels: {test_counts}")

                    results["test"]["ACC"].append(acc)
                    results["test"]["bACC"].append(bacc)
                    results["test"]["F1"].append(f1)
                    results["test"]["AUC"].append(auc)

                logging.info(f"Average ACC | bACC | F1 | AUC: {np.mean(results["test"]["ACC"])} {np.mean(results["test"]["bACC"])} {np.mean(results["test"]["F1"])} {np.mean(results["test"]["AUC"])}")
                RESULTS_FILE = os.path.join(SSL_ROOT, "results", "downstream", "classification", f"{model_type}_{mask_pct}_{classifier}_{method}_{label_type}.json")
                with open(RESULTS_FILE, "w") as json_file:
                    json.dump(results, json_file, indent=4)

            # Regression ------------------------------------------------------------
            all_labels = generate_continuous_labels_day(subject_ids, dates, label_types=continuous_labels)
            for label_type in continuous_labels:
                logging.info("-"*50)
                logging.info(f"Label: {label_type}")
                logging.info("-"*50)
                logging.info(f"Starting at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")

                labels = all_labels[label_type]
                nan_indices = [i for i in range(len(labels)) if np.isnan(labels[i])]
                nan_indices.sort(reverse=True)
                subject_ids_copy = copy.deepcopy(subject_ids)
                data_copy = copy.deepcopy(data)
                for i in nan_indices:
                    subject_ids_copy.pop(i)
                    data_copy.pop(i)
                    labels.pop(i)

                subject_ids_copy = np.asarray(subject_ids_copy)
                data_copy = np.asarray(data_copy)
                labels = np.asarray(labels)

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

                for i, (train_index, test_index) in enumerate(idxs):
                    train_subject_ids = subject_ids_copy[train_index].tolist()
                    train_data = []
                    for idx in train_index: train_data.append(data_copy[idx])
                    train_labels = labels[train_index]
                    train_counts = Counter(train_labels)
                    train_counts = list(train_counts.items())
                    train_labels = train_labels.tolist()

                    test_subject_ids = subject_ids_copy[test_index].tolist()
                    test_data = []
                    for idx in test_index: test_data.append(data_copy[idx])
                    test_labels = labels[test_index]
                    test_counts = Counter(test_labels)
                    test_counts = list(test_counts.items())
                    test_labels = test_labels.tolist()

                    train_dataset = TilesDataset(train_subject_ids, train_data, train_labels)
                    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True, generator=torch.Generator(device=device))
                    test_dataset = TilesDataset(test_subject_ids, test_data, test_labels)
                    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, shuffle=False, generator=torch.Generator(device=device))

                    results["splits"]["train_size"].append(len(train_labels))
                    results["splits"]["test_size"].append(len(test_labels))
                    logging.info(f"Training on {len(train_labels)} samples, testing on {len(test_labels)}.")

                    model = LinearHead(d_input=d_input, sequence_len=sequence_len)

                    optimizer = optim.AdamW(
                        model.parameters(),
                        # lr=5e-3,
                        lr=1e-3,
                        # weight_decay=1e-4,
                        # betas=(0.9, 0.95)
                    )

                    for epoch in range(epochs):
                        mse, mae, pearsonr_result = train_epoch(
                            train_dataloader, model, device, optimizer, eval_type="regression"
                        )
                        r = pearsonr_result[0]
                        p = pearsonr_result[1]
                        results["train"]["MSE"].append(mse)
                        results["train"]["MAE"].append(mae)
                        results["train"]["R"].append(r)
                        results["train"]["p"].append(p)
                    
                    mse, mae, pearsonr_result = validate_epoch(
                        test_dataloader, model, device, eval_type="regression"
                    )
                    r = pearsonr_result[0]
                    p = pearsonr_result[1]
                    results["test"]["MSE"].append(mse)
                    results["test"]["MAE"].append(mae)
                    results["test"]["R"].append(r)
                    results["test"]["p"].append(p)

                logging.info(f"Average MSE | MAE | R | p: {np.mean(results["test"]["MSE"])} {np.mean(results["test"]["MAE"])} {np.mean(results["test"]["R"])} {np.mean(results["test"]["p"])}")
                RESULTS_FILE = os.path.join(SSL_ROOT, "results", "downstream", "regression", f"{model_type}_{mask_pct}_{classifier}_{method}_{label_type}.json")
                with open(RESULTS_FILE, "w") as json_file:
                    json.dump(results, json_file, indent=4)



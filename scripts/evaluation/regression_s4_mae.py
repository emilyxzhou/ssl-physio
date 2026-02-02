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

import argparse
import copy
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import constants

from collections import Counter
from datetime import datetime
from imblearn.over_sampling import RandomOverSampler
from pathlib import Path
from scipy.stats import pearsonr, ConstantInputWarning
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GroupKFold
from torch import optim
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

from s4_mae import S4MAE
from trainer import Trainer
from tiles_dataloader import get_data_from_splits, TilesDataset, generate_binary_labels, generate_continuous_labels_day

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
    format="%(message)s", 
    level=logging.INFO, 
    # datefmt="%Y-%m-%d %H:%M:%S"
)

os.environ["S4_FAST_CAUCHY"] = "0"
os.environ["S4_FAST_VAND"] = "0"
os.environ["S4_BACKEND"] = "keops"   # or "keops" if you installed pykeops

SSL_ROOT = os.path.join(USER_ROOT, "ssl-physio")


def load_model(checkpoint_path, model_params, classification=False, device="cuda:1"):
    # Read arguments -----------------------------------------------------------------------------------------------
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model = S4MAE(
        **model_params,
        classification=classification,
        device=device
    ).to(device)

    # Load weights
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model
    

def freeze_weights(model):
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model


def train_epoch(
    dataloader, 
    model, 
    device, 
    optimizer,
    epoch
):
    model.eval()
    model.cls_head.train()
    criterion = nn.L1Loss()
    total_labels = []
    total_preds = []
        
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

        preds = preds.detach().cpu().numpy().astype(float)
        labels = labels.detach().cpu().numpy().astype(float)
        total_labels.extend(labels)
        total_preds.extend(preds)

        mse = mean_squared_error(total_labels, total_preds)
        mae = mean_absolute_error(total_labels, total_preds)
        pearsonr_result = pearsonr(total_labels, total_preds)
        
        # if (epoch+1 % 10 == 0) and (batch_idx+1 % 50 == 0 or batch_idx+1 == len(dataloader)):
        # if (batch_idx+1 % 50 == 0 or batch_idx+1 == len(dataloader)):
        #     logging.info(f"Current MSE at epoch {epoch+1}, step {batch_idx+1}/{len(dataloader)} {mse}")
        #     logging.info(f"Current MAE at epoch {epoch+1}, step {batch_idx+1}/{len(dataloader)} {mae}")
        #     logging.info(f"Current R at epoch {epoch+1}, step {batch_idx+1}/{len(dataloader)} {pearsonr_result}")
        #     logging.info(f"------------------------------------------------------------")

    return mse, mae, pearsonr_result


def validate_epoch(
    dataloader, 
    model, 
    device,
    epoch
):
    model.eval()
    model.cls_head.eval()
    criterion = nn.L1Loss()
    total_labels = []
    total_preds = []
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            # Transfer to GPU
            batch, subject_ids, labels = batch_data
            batch = batch.to(device)                  # original shape (batch_size, 1440, num_features)
            batch = torch.transpose(batch, 1, 2)    # new shape (batch_size, num_features, 1440)

            preds = model(batch).flatten()
            loss = criterion(preds.flatten().float(), labels.float()).to(device)
            preds = preds.to(device)
            labels = labels.to(device)

            preds = preds.detach().cpu().numpy().astype(float)
            labels = labels.detach().cpu().numpy().astype(float)
            total_labels.extend(labels)
            total_preds.extend(preds)

            mse = mean_squared_error(total_labels, total_preds)
            mae = mean_absolute_error(total_labels, total_preds)
            pearsonr_result = pearsonr(total_labels, total_preds)
            
            # if (epoch+1 % 10 == 0) and (batch_idx+1 % 50 == 0 or batch_idx+1 == len(dataloader)):
            if (batch_idx+1 % 50 == 0 or batch_idx+1 == len(dataloader)):
                # logging.info(f"Current MSE at epoch {epoch+1}, step {batch_idx+1}/{len(dataloader)} {mse}")
                # logging.info(f"Current MAE at epoch {epoch+1}, step {batch_idx+1}/{len(dataloader)} {mae}")
                # logging.info(f"Current R at epoch {epoch+1}, step {batch_idx+1}/{len(dataloader)} {r}")
                logging.info(f"MSE | MAE | R: {mse} {mae} {pearsonr_result}")
                logging.info(f"------------------------------------------------------------")
        return mse, mae, pearsonr_result


if __name__ == "__main__":
    # Read arguments -----------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Script for S4-MAE downstream regression.")
    parser.add_argument("--debug", "-d", type=str, default=False)
    parser.add_argument("--classification", "-c", type=str, default="finetune")
    parser.add_argument("--device", "-dev", type=str, default="cuda:1")
    args = parser.parse_args()
    debug = args.debug
    classification = args.classification
    unfreeze_seq = (classification != "lin_probe")
    device = args.device

    # Find device
    device = torch.device(device) if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.set_default_device(device)
        print("Default device set to CUDA.")
    else:
        print("CUDA not available.")
    # torch.set_default_dtype(torch.float64)

    config_path = os.path.join(SSL_ROOT, "config", "s4_config.json")
    config = json.load(open(config_path, "r"))
    training_params = config["training_params"]
    model_params = config["model_params"]

    for mask_pct in [10, 30, 50, 70]:
        logging.info("="*60)
        logging.info(f"{mask_pct}% masking")
        logging.info("="*60)
        MODEL_SAVE_PATH = f"{USER_ROOT}/ssl-physio/models/reconstruction/s4-mae_{mask_pct}.pt"

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
            constants.Labels.RHR,
            constants.Labels.STEPS,
            constants.Labels.SLEEP_MINS
        ]

        subject_ids, dates, data = get_data_from_splits()
        all_labels = generate_continuous_labels_day(subject_ids, dates, label_types=label_types)
        
        for label_type in label_types:
            logging.info("-"*60)
            logging.info(f"{label_type}")
            logging.info("-"*60)
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

            group_kfold = GroupKFold(n_splits=5, shuffle=True, random_state=13)

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

                train_dataset = TilesDataset(train_subject_ids, train_data, train_labels)
                train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True, generator=torch.Generator(device=device))
                test_dataset = TilesDataset(test_subject_ids, test_data, test_labels)
                test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, shuffle=False, generator=torch.Generator(device=device))

                results["splits"]["train_size"].append(len(train_labels))
                results["splits"]["test_size"].append(len(test_labels))
                logging.info(f"Training on {len(train_labels)} samples, testing on {len(test_labels)}.")

                # Load model ------------------------------------------------------------------------------------------------
                model = load_model(MODEL_SAVE_PATH, model_params, classification=classification, device=device)
                model = freeze_weights(model)

                unfreeze_seq = False
                if unfreeze_seq:
                    for layer in model.seq_model.s4_layers[-1:]:
                        for param in layer.parameters():
                            param.requires_grad = True
                        
                for param in model.cls_head.parameters():
                    param.requires_grad = True

                if i == 0: summary(model)

                optimizer = optim.AdamW(
                    model.parameters(),
                    # lr=5e-3,
                    lr=1e-3,
                    # weight_decay=1e-4,
                    # betas=(0.9, 0.95)
                )

                for epoch in range(epochs):
                    mse, mae, pearsonr_result = train_epoch(
                        train_dataloader, model, device, optimizer, epoch=epoch
                    )
                    r = pearsonr_result[0]
                    p = pearsonr_result[1]
                    results["train"]["MSE"].append(mse)
                    results["train"]["MAE"].append(mae)
                    results["train"]["R"].append(r)
                    results["train"]["p"].append(p)
                    
                mse, mae, pearsonr_result = validate_epoch(
                    test_dataloader, model, device, epoch=0
                )
                r = pearsonr_result[0]
                p = pearsonr_result[1]
                results["test"]["MSE"].append(mse)
                results["test"]["MAE"].append(mae)
                results["test"]["R"].append(r)
                results["test"]["p"].append(p)

            logging.info(f"Average MSE | MAE | R | p: {np.mean(results["test"]["MSE"])} {np.mean(results["test"]["MAE"])} {np.mean(results["test"]["R"])} {np.mean(results["test"]["p"])}")
            # RESULTS_FILE = os.path.join(SSL_ROOT, "results", "downstream", "regression", f"s4_{mask_pct}_{classification}_{label_type}.json")
            RESULTS_FILE = os.path.join(SSL_ROOT, "results", "downstream", "regression", f"s4_{mask_pct}_conv_probe_{label_type}.json")
            with open(RESULTS_FILE, "w") as json_file:
                json.dump(results, json_file, indent=4)

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
import pandas as pd
import pprint
import random
import signal
import time
import torch
import torch.nn as nn
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

from s4_mae import S4MAE

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

MODEL_SAVE_PATH = f"{USER_ROOT}/ssl-physio/models/reconstruction/s4-mae_2025-12-22_11:15:39.pt"    # 10% masking
# MODEL_SAVE_PATH = f"{USER_ROOT}/ssl-physio/models/reconstruction/s4-mae_2025-12-23_06:39:27.pt"    # 30% masking
# MODEL_SAVE_PATH = f"{USER_ROOT}/ssl-physio/models/reconstruction/s4-mae_2026-01-04_21:48:55.pt"    # 50% masking

classification = "lin_probe"
unfreeze_s4 = (classification != "lin_probe")
mask_ratio = 0.1


def load_model(checkpoint_path, classification=False, verbose=False):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model = S4MAE(
        d_model=dec_hidden_dims[0],
        d_input=d_input,
        d_output=d_output,
        enc_hidden_dims=enc_hidden_dims,
        dec_hidden_dims=dec_hidden_dims,
        n_layers_s4=n_layers_s4,
        mask_ratio=mask_ratio,
        classification=classification,
        verbose=verbose
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
    # Read arguments -----------------------------------------------------------------------------------------------
    with open("/home/emilyzho/ssl-physio/scripts/params.yaml", "r") as file:
        params = yaml.safe_load(file)
        mode = params["mode"]
        reconstruction = params["reconstruction"]
        scale = params["scale"]
        d_input = params["d_input"]
        d_output = params["d_output"]
        enc_hidden_dims = params["enc_hidden_dims"]
        dec_hidden_dims = params["dec_hidden_dims"]
        d_model = params["d_model"]
        n_layers_s4 = params["n_layers_s4"]
        mask_ratio = params["mask_ratio"]
        lr = params["lr"]
    mask_ratio = 0.5
    if dec_hidden_dims is not None: d_model = dec_hidden_dims[0]

    parser = argparse.ArgumentParser(description="Script for S4-MAE pre-training.")
    parser.add_argument("--mode", "-m", type=str, default="full")
    parser.add_argument("--reconstruction", "-r", type=str, default="full")
    parser.add_argument("--debug", "-d", type=str, default=False)
    args = parser.parse_args()
    mode = args.mode
    debug = args.debug
    reconstruction = args.reconstruction

    pprint.pprint(params)

    classification = "lin_probe"
    unfreeze_s4 = (classification != "lin_probe")

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
    epochs = 50
    num_training_samples = 5    # First n days for each subject
    debug = True

    # Loading data -----------------------------------------------------------------------------------------------
    signal_columns = [
        # "RMSStdDev_ms", "RRPeakCoverage", "SDNN_ms", "RR0", 
        # "sleepId", "level", 
        "bpm", "StepCount"
    ]
    scale = "mean"
    window_size = 15    # minutes
    label_types = [
        "anxiety", 
        # "stress"
    ]

    for label_type in label_types:
        logging.info(f"Label: {label_type} " + "-"*80)

        subject_ids, dates, data = load_tiles_holdout(
            signal_columns=signal_columns,
            scale=scale, window_size=window_size, debug=debug
        )
        labels = generate_binary_labels(subject_ids, dates, version="holdout", label_type=label_type)
        unique_subjects = np.unique(subject_ids).tolist()

        accs = list()
        baccs = list()
        f1s = list()
        aucs = list()

        df = pd.DataFrame({"ID": subject_ids, "Date": dates, label_type: labels})
        for subject_id in unique_subjects[0:1]:
            subset = df[df["ID"] == subject_id]
            subset = subset.sort_values(by="Date")

            train_labels = subset.iloc[0:num_training_samples, df.columns.get_loc(label_type)]
            test_labels = subset.iloc[num_training_samples:, df.columns.get_loc(label_type)]

            train_data = []
            test_data = []

            train_indices = train_labels.index.to_list()
            for i in subset.index.tolist():
                if i in train_indices:
                    train_data.append(data[i])
                else:
                    test_data.append(data[i])

            train_subject_ids = [subject_id for _ in range(len(train_data))]
            test_subject_ids = [subject_id for _ in range(len(test_data))]

            tiles_holdout_train = TilesDataset(
                subject_ids=train_subject_ids, data=train_data, labels=train_labels
            )
            train_dataloader = DataLoader(tiles_holdout_train, batch_size=num_training_samples, num_workers=0, shuffle=True, generator=torch.Generator(device=device))

            tiles_holdout_test = TilesDataset(
                subject_ids=test_subject_ids, data=test_data, labels=test_labels
            )
            test_dataloader = DataLoader(tiles_holdout_test, batch_size=num_training_samples, num_workers=0, shuffle=False)

            model = load_model(MODEL_SAVE_PATH, classification=classification)
            model = freeze_weights(model)

            if unfreeze_s4:
                for layer in model.s4_model.s4_layers[-1:]:
                    for param in layer.parameters():
                        param.requires_grad = True
                    
            for param in model.cls_head.parameters():
                param.requires_grad = True

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

            accs.append(acc)
            baccs.append(bacc)
            f1s.append(f1)
            aucs.append(auc)

        logging.info(f"Average ACC | bACC | F1 | AUC: {np.mean(accs)} {np.mean(baccs)} {np.mean(f1s)} {np.mean(aucs)}")

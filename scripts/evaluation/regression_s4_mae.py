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
import wandb
import yaml

import constants

from collections import Counter
from datetime import datetime
from imblearn.over_sampling import RandomOverSampler
from pathlib import Path
from scipy.stats import pearsonr, ConstantInputWarning
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

from trainer import Trainer, split_k_fold
from tiles_dataloader import load_tiles_open, load_tiles_holdout, TilesDataset, generate_binary_labels, generate_continuous_labels_day

from s4_mae import S4MAE

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", category=ConstantInputWarning, message="An input array is constant; the correlation coefficient is not defined.")

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

# MODEL_SAVE_PATH = f"{USER_ROOT}/ssl-physio/models/reconstruction/s4-mae_2025-12-22_11:15:39.pt"    # 10% masking
MODEL_SAVE_PATH = f"{USER_ROOT}/ssl-physio/models/reconstruction/s4-mae_2025-12-23_06:39:27.pt"    # 30% masking
# MODEL_SAVE_PATH = f"{USER_ROOT}/ssl-physio/models/reconstruction/s4-mae_2026-01-04_21:48:55.pt"    # 50% masking

classification = "lin_probe"
unfreeze_s4 = (classification != "lin_probe")
mask_ratio = 0.3


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

        preds = preds.detach().cpu().numpy().astype(int)
        labels = labels.detach().cpu().numpy().astype(int)
        total_labels.extend(labels)
        total_preds.extend(preds)

        mse = mean_squared_error(total_labels, total_preds)
        mae = mean_absolute_error(total_labels, total_preds)
        r = pearsonr(total_labels, total_preds)
        
        # if (epoch+1 % 10 == 0) and (batch_idx+1 % 50 == 0 or batch_idx+1 == len(dataloader)):
        # if (batch_idx+1 % 50 == 0 or batch_idx+1 == len(dataloader)):
        #     logging.info(f"Current MSE at epoch {epoch+1}, step {batch_idx+1}/{len(dataloader)} {mse}")
        #     logging.info(f"Current MAE at epoch {epoch+1}, step {batch_idx+1}/{len(dataloader)} {mae}")
        #     logging.info(f"Current R at epoch {epoch+1}, step {batch_idx+1}/{len(dataloader)} {r}")
        #     logging.info(f"------------------------------------------------------------")

    return mse, mae, r


def validate_epoch(
    dataloader, 
    model, 
    device,
    epoch,
    split:  str="test"
):
    model.eval()
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
            r = pearsonr(total_labels, total_preds)
            
            # if (epoch+1 % 10 == 0) and (batch_idx+1 % 50 == 0 or batch_idx+1 == len(dataloader)):
            if (batch_idx+1 % 50 == 0 or batch_idx+1 == len(dataloader)):
                # logging.info(f"Current MSE at epoch {epoch+1}, step {batch_idx+1}/{len(dataloader)} {mse}")
                # logging.info(f"Current MAE at epoch {epoch+1}, step {batch_idx+1}/{len(dataloader)} {mae}")
                # logging.info(f"Current R at epoch {epoch+1}, step {batch_idx+1}/{len(dataloader)} {r}")
                logging.info(f"MSE | MAE | R: {mse} {mae} {r}")
                logging.info(f"------------------------------------------------------------")
        return mse, mae, r


if __name__ == "__main__":
    # Read arguments -----------------------------------------------------------------------------------------------
    with open("/home/emilyzho/ssl-physio/scripts/params_s4.yaml", "r") as file:
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
        # mask_ratio = params["mask_ratio"]
        lr = params["lr"]
    if dec_hidden_dims is not None: d_model = dec_hidden_dims[0]

    parser = argparse.ArgumentParser(description="Script for S4-MAE regression evaluation.")
    parser.add_argument("--mode", "-m", type=str, default="full")
    parser.add_argument("--reconstruction", "-r", type=str, default="full")
    parser.add_argument("--debug", "-d", type=str, default=False)
    args = parser.parse_args()
    mode = args.mode
    debug = args.debug
    reconstruction = args.reconstruction

    pprint.pprint(params)

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
        constants.Labels.STEPS,
        constants.Labels.HR,
        constants.Labels.SDNN
    ]
    num_folds = 5
    for label_type in label_types:
        logging.info(f"Label: {label_type} " + "-"*80)

        subject_ids, dates, data = load_tiles_holdout(
            signal_columns=signal_columns,
            scale=scale, window_size=window_size, debug=debug
        )
        labels = generate_continuous_labels_day(subject_ids, dates, version="holdout", label_type=label_type, debug=debug)
        if label_type == constants.Labels.STEPS:
            labels[:] = [s / 1000 for s in labels]
        nan_indices = [i for i in range(len(labels)) if np.isnan(labels[i])]
        nan_indices.sort(reverse=True)
        for i in nan_indices:
            subject_ids.pop(i)
            data.pop(i)
            labels.pop(i)

        # 5 folds, randomly sampling 700 samples as the training set and using the remaining as the test set
        # subject_id_folds, data_folds, labels_folds = split_k_fold(subject_ids, data, labels, num_folds=num_folds, seed=37)
        mses = list()
        maes = list()
        rs = list()

        for i in range(num_folds):
            logging.info(f"Fold {i+1}")
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
            num_train_samples = int(len(subject_ids) * 0.14)
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
                
            mse, mae, r = validate_epoch(
                test_dataloader, model, device, epoch=0, 
                split="test"
            )

            mses.append(mse)
            maes.append(mae)
            rs.append(r)

        logging.info(f"\nAverage MSE: {np.mean(mses)}")
        logging.info(f"\nAverage MAE: {np.mean(maes)}")
        logging.info(f"\nAverage R: {np.mean(rs)}")


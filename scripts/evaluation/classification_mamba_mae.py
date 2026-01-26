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

import argparse
import json
import logging
import numpy as np
import pandas as pd
import pprint
import torch
import torch.nn as nn
import yaml

import constants

from collections import Counter
from datetime import datetime
from imblearn.over_sampling import RandomOverSampler
from pathlib import Path
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, roc_auc_score
from torch import optim
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

from mamba_mae import MambaMAE
from trainer import Trainer
from tiles_dataloader import get_pretrain_eval_dataloaders
from utils import stratified_group_split, get_kfold_loaders


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

SSL_ROOT = os.path.join(USER_ROOT, "ssl-physio")


def load_model(checkpoint_path, config_path, classification=False, mask_ratio=None):
    # Read arguments -----------------------------------------------------------------------------------------------
    config = json.load(open(config_path, "r"))
    model_params = config["model_params"]
    if model_params["dec_hidden_dims"] is not None: model_params["d_model"] = model_params["dec_hidden_dims"][0]
    if mask_ratio is not None: model_params["mask_ratio"] = mask_ratio

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model = MambaMAE(
        **model_params,
        classification=classification
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
    parser = argparse.ArgumentParser(description="Script for S4-MAE downstream classification.")
    parser.add_argument("--debug", "-d", type=str, default=False)
    parser.add_argument("--classification", "-c", type=str, default="lin_probe")
    args = parser.parse_args()
    debug = args.debug
    classification = args.classification
    unfreeze_seq = (classification != "lin_probe")

    # Find device
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.set_default_device("cuda:0")
        print("Default device set to CUDA.")
    else:
        print("CUDA not available.")
    # torch.set_default_dtype(torch.float64)

    config_path = os.path.join(SSL_ROOT, "config", "mamba_config.json")
    config = json.load(open(config_path, "r"))
    training_params = config["training_params"]
    model_params = config["model_params"]

    for mask_ratio in [0.1, 0.3, 0.5, 0.7]:
        logging.info(f"Mask ratio: {mask_ratio} " + "-"*120)
        MODEL_SAVE_PATH = f"{USER_ROOT}/ssl-physio/models/reconstruction/s4-mae_{int(mask_ratio*100)}.pt"

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

            _, _, test_dataloader = get_pretrain_eval_dataloaders(
                signal_columns, label_type=label_type,
                scale="mean", window_size=15, 
                batch_size=32, train_test_split=0.9,
                device=device, debug=debug,
                random_seed=42
            )

            folds = get_kfold_loaders(test_dataloader)

            for fold_idx, (train_dataloader, test_dataloader) in enumerate(folds):

                # train_arr = np.array(train_dataloader.dataset.labels)
                # train_counts = Counter(train_arr)
                # train_counts = list(train_counts.items())
                # test_arr = np.array(test_dataloader.dataset.labels)
                # test_counts = Counter(test_arr)
                # test_counts = list(test_counts.items())

                # results["splits"]["train_size"].append(train_arr.size)
                # results["splits"]["test_size"].append(test_arr.size)
                # results["splits"]["train_labels"][0].append(train_counts[0][1])
                # results["splits"]["train_labels"][1].append(train_counts[1][1])
                # results["splits"]["test_labels"][0].append(test_counts[0][1])
                # results["splits"]["test_labels"][1].append(test_counts[1][1])

                num_train_samples = len(train_dataloader)
                num_test_samples = len(test_dataloader)

                # Load model ------------------------------------------------------------------------------------------------
                model = load_model(MODEL_SAVE_PATH, config_path, classification=classification, mask_ratio=mask_ratio)
                model = freeze_weights(model)

                if unfreeze_seq:
                    for layer in model.seq_model.mamba_layers[-1:]:
                        for param in layer.parameters():
                            param.requires_grad = True
                        
                for param in model.cls_head.parameters():
                    param.requires_grad = True

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

                # logging.info(f"Training labels: {train_counts}")
                # logging.info(f"Testing labels: {test_counts}")

                results["test"]["ACC"].append(acc)
                results["test"]["bACC"].append(bacc)
                results["test"]["F1"].append(f1)
                results["test"]["AUC"].append(auc)

            logging.info(f"Average ACC | bACC | F1 | AUC: {np.mean(results["test"]["ACC"])} {np.mean(results["test"]["bACC"])} {np.mean(results["test"]["F1"])} {np.mean(results["test"]["AUC"])}")
            RESULTS_FILE = os.path.join(SSL_ROOT, "results", "downstream", "classification", f"mamba_{int(mask_ratio*100)}_{classification}_{label_type}.json")
            with open(RESULTS_FILE, "w") as json_file:
                json.dump(results, json_file, indent=4)

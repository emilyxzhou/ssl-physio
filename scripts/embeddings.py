import os
import sys
from pathlib import Path
USER_ROOT = str(Path(__file__).resolve().parents[2])
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

mask_ratio = 0.3


def load_model(checkpoint_path, verbose=False):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model = S4MAE(
        d_model=dec_hidden_dims[0],
        d_input=d_input,
        d_output=d_output,
        enc_hidden_dims=enc_hidden_dims,
        dec_hidden_dims=dec_hidden_dims,
        n_layers_s4=n_layers_s4,
        mask_ratio=mask_ratio,
        classification=False,
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
        # mask_ratio = params["mask_ratio"]
        lr = params["lr"]
    if dec_hidden_dims is not None: d_model = dec_hidden_dims[0]

    parser = argparse.ArgumentParser(description="Script for generating embeddings.")
    parser.add_argument("--mode", "-m", type=str, default="full")
    parser.add_argument("--debug", "-d", type=str, default=False)
    args = parser.parse_args()
    mode = args.mode
    debug = args.debug

    pprint.pprint(params)

    # Find device
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.set_default_device("cuda:0")
        print("Default device set to CUDA.")
    else:
        print("CUDA not available.")
    torch.set_default_dtype(torch.float64)

    # Loading data -----------------------------------------------------------------------------------------------
    signal_columns = [
        # "RMSStdDev_ms", "RRPeakCoverage", "SDNN_ms", "RR0", 
        # "sleepId", "level", 
        "bpm", "StepCount"
    ]
    scale = "mean"
    window_size = 15    # minutes
    subject_ids, dates, data = load_tiles_holdout(
        signal_columns=signal_columns,
        scale=scale, window_size=window_size, debug=debug
    )

    # NOTES:
    # debug = True loads data for five subjects only
    # subject_ids = list of subject IDs
    # dates = list of datetime date objects
    # data = list of numpy arrays (2, 1440), where the first column is heart rate and the second column is step count

    # Load model ------------------------------------------------------------------------------------------------
    model = load_model(MODEL_SAVE_PATH)
    model = freeze_weights(model)

    # model.eval()
    # out, target, mask = model(x, mask_ratio=mask_ratio)
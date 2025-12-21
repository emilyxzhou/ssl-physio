import os
import sys
from pathlib import Path
root = str(Path(__file__).resolve().parents[3])
paths = [
    os.path.join(
        root, "ssl-physio", "src", "dataloaders"
    ),
    os.path.join(
        root, "ssl-physio", "src", "s4-models"
    ),
    os.path.join(
        root, "ssl-physio", "src", "trainers"
    )
]
for path in paths:
    sys.path.insert(0, path)

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

from datetime import datetime
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

from trainer import Trainer
from tiles_dataloader import load_tiles_open, load_tiles_holdout, TilesDataset

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

# Save paths 
MODEL_SAVE_FOLDER = "/home/emilyzho/ssl-physio/models/reconstruction"

CHECKPOINT_DIR = "/home/emilyzho/ssl-physio/ckpts"
CHECKPOINT_PREFIX = "S4"


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
    if dec_hidden_dims is not None: d_model = dec_hidden_dims[0]

    parser = argparse.ArgumentParser(description="Script for S4-MAE pre-training.")
    parser.add_argument("--mode", "-m", type=str, default="full")
    parser.add_argument("--reconstruction", "-r", type=str, default="full")
    parser.add_argument("--debug", "-d", type=str, default=True)
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
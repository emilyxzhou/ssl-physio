import os
import sys
from pathlib import Path
USER_ROOT = str(Path(__file__).resolve().parents[3])
paths = [
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

import argparse
import json
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
from tiles_dataloader import get_pretrain_eval_dataloaders

from mamba_mae import MambaMAE

# Define logging console
import logging
logging.basicConfig(
    format="%(asctime)s %(levelname)-3s ==> %(message)s", 
    level=logging.INFO, 
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Paths 
SSL_ROOT = os.path.join(USER_ROOT, "ssl-physio")
MODEL_SAVE_FOLDER = f"{USER_ROOT}/ssl-physio/models/reconstruction"
CHECKPOINT_DIR = f"{USER_ROOT}/ssl-physio/ckpts"
CHECKPOINT_PREFIX = "mamba"

def save_model(model, save_path):
    torch.save(
        model.state_dict(), 
        save_path
    )
    print(f"Model saved as {save_path}. Exiting.")
    if use_wandb:
        artifact = wandb.Artifact("mamba-mae", type="model")
        artifact.add_file(save_path)
        wandb.log_artifact(artifact)
        wandb.finish()


if __name__ == "__main__":
    # Read arguments -----------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Script for Mamba-MAE pre-training.")
    parser.add_argument("--debug", "-d", action="store_true", default=False)
    parser.add_argument("--mask_ratio", "-m", type=float, default=None)
    args = parser.parse_args()
    debug = args.debug
    mask_ratio = args.mask_ratio

    config_path = os.path.join(SSL_ROOT, "config", "mamba_config.json")
    config = json.load(open(config_path, "r"))

    training_params = config["training_params"]

    model_params = config["model_params"]
    if model_params["dec_hidden_dims"] is not None: model_params["d_model"] = model_params["dec_hidden_dims"][0]
    if mask_ratio is not None: model_params["mask_ratio"] = mask_ratio

    pprint.pprint(model_params)

    # Find device
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.set_default_device("cuda:0")
        print("Default device set to CUDA.")
    else:
        print("CUDA not available.")
    # torch.set_default_dtype(torch.float32)

    # Define parameters ----------------------------------------------------------------------------------------------------
    use_wandb = not debug
    verbose = False

    # Save network
    # Save network
    MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_FOLDER, f"mamba-mae_{int(model_params['mask_ratio']*100)}.pt")

    # Training variables
    epochs = training_params["epochs"]
    batch_size = training_params["batch_size"]

    # Initialize wandb ----------------------------------------------------------------------------------------------------
    if use_wandb:
        wandb.init(
            project="ssl-s4",
            name=f"Mamba-{int(model_params['mask_ratio']*100)}",
            config = {
                "scaling": training_params["scaling"],
                "epochs": training_params["epochs"],
                "optimizer": "AdamW",
                "encoder": model_params["enc_hidden_dims"],
                "decoder": model_params["dec_hidden_dims"],
                "Mamba dim": model_params["d_model"],
                "# Mamba layers": model_params["n_layers_seq"],
                "d_output": model_params["d_output"],
                "masking ratio": model_params["mask_ratio"],
                "learning rate": training_params["lr"],
                "mode": training_params["mode"],
                "reconstruction": training_params["reconstruction"],
                "save path": MODEL_SAVE_PATH
            }
        )

    # Setting up models -----------------------------------------------------------------------------------------------
    model = MambaMAE(
        **model_params,
        classification=False
    ).to(device)
    summary(model, input_size=(1, model_params["d_input"], 1440))

    # Loading data -----------------------------------------------------------------------------------------------
    signal_columns = [
        # "RMSStdDev_ms", "RRPeakCoverage", "SDNN_ms", "RR0", 
        # "sleepId", "level", 
        "bpm", "StepCount"
    ]
    scale = training_params["scaling"]
    window_size = training_params["window_size"]    # Window size for moving average 
    label_type = None

    pretrain_dataloader, val_dataloader, test_dataloader = get_pretrain_eval_dataloaders(
        signal_columns, label_type=label_type,
        scale="mean", window_size=15, 
        batch_size=32, train_test_split=0.9,
        device=device, debug=debug,
        random_seed=42
    )

    num_train_samples = len(pretrain_dataloader.dataset)
    steps_per_train_epoch = math.ceil(num_train_samples / batch_size)
    total_train_steps = epochs * steps_per_train_epoch
    warmup_steps = int(0.05 * total_train_steps)    # 5% warmup

    # Model training ------------------------------------------------------------------------------------------------
    # Define learning rate
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            # linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        # cosine decay after warmup
        progress = float(current_step - warmup_steps) / float(max(1, total_train_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))   # cosine from 1 → 0

    # Define optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_params["lr"],
        weight_decay=training_params["weight_decay"],
        betas=(0.9, 0.95)
    )
    scheduler = None

    # Read trainable params
    model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    params = sum([np.prod(p.size()) for p in model_parameters])
    logging.info(f"Trainable params: {params/(1e6):.2f} M")

    trainer = Trainer(
        n_epochs=epochs, 
        checkpoint_dir=CHECKPOINT_DIR, checkpoint_prefix=CHECKPOINT_PREFIX, 
        model_save_folder=MODEL_SAVE_FOLDER,
        mode=training_params["mode"], reconstruction=training_params["reconstruction"], use_wandb=use_wandb
    )
    
    start_datetime = datetime.now()
    start_str = start_datetime.strftime("%Y-%m-%d_%H:%M:%S")
    logging.info(f"Training start: {start_str}.")

    model = trainer.train_recon(
        model, pretrain_dataloader, val_dataloader, criterion=nn.MSELoss(),
        optimizer=optimizer, scheduler=scheduler, mask_ratio=model_params["mask_ratio"],
        resume_checkpoint=None, device=device, debug=debug
    )

    if not debug:
        test_loss = trainer.validate_recon(
            model, test_dataloader, criterion=nn.MSELoss(),
            mask_ratio=model_params["mask_ratio"], split="test",
            device=device
        )
    
    end_datetime = datetime.now()
    end_str = end_datetime.strftime("%Y-%m-%d_%H:%M:%S")
    total_seconds = (end_datetime - start_datetime).total_seconds()
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)

    logging.info(f"Training end: {end_str}. Elapsed time: {minutes}:{seconds}.")

    # Save model
    if not debug: save_model(model, MODEL_SAVE_PATH)
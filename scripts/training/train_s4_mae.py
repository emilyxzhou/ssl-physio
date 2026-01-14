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
MODEL_SAVE_FOLDER = f"{USER_ROOT}/ssl-physio/models/reconstruction"

CHECKPOINT_DIR = f"{USER_ROOT}/ssl-physio/ckpts"
CHECKPOINT_PREFIX = "S4"


# Define loop functions -----------------------------------------------------------------------------------------------
def save_model(model):
    torch.save(
        model.state_dict(), 
        MODEL_SAVE_PATH
    )
    print(f"Model saved as {MODEL_SAVE_PATH}. Exiting.")
    if use_wandb:
        artifact = wandb.Artifact("s4-mae", type="model")
        artifact.add_file(MODEL_SAVE_PATH)
        wandb.log_artifact(artifact)
        wandb.finish()


# Set up exit handler -------------------------------------------------------------------------------------------------
def signal_handler(sig, frame):
    print("\nCtrl+C detected.")
    if not debug: 
        save_model(model)
        print("Saving model...")
    exit(0)


if __name__ == "__main__":
    # Read arguments -----------------------------------------------------------------------------------------------
    with open(f"{USER_ROOT}/ssl-physio/scripts/params_s4.yaml", "r") as file:
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

    # Find device
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.set_default_device("cuda:0")
        print("Default device set to CUDA.")
    else:
        print("CUDA not available.")
    torch.set_default_dtype(torch.float64)


    # Define parameters ----------------------------------------------------------------------------------------------------
    use_wandb = not debug
    verbose = False

    # Save network
    START_DATETIME = datetime.now()
    START_DATETIME = START_DATETIME.strftime("%Y-%m-%d_%H:%M:%S")
    MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_FOLDER, f"s4-mae_{START_DATETIME}.pt")

    # Training variables
    epochs = 50
    batch_size = 32


    # Initialize wandb ----------------------------------------------------------------------------------------------------
    if use_wandb:
        wandb.init(
            project="ssl-s4",
            name="S4",
            config = {
                "scaling": scale,
                "epochs": epochs,
                "optimizer": "AdamW",
                "encoder": enc_hidden_dims,
                "decoder": dec_hidden_dims,
                "S4 dim": d_model,
                "# S4 layers": n_layers_s4,
                "d_output": d_output,
                "masking ratio": mask_ratio,
                "learning rate": lr,
                "mode": mode,
                "reconstruction": reconstruction,
                "save path": MODEL_SAVE_PATH
            }
        )

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


    # Setting up models -----------------------------------------------------------------------------------------------
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
    summary(model, input_size=(1, d_input, 1440))


    # Loading data -----------------------------------------------------------------------------------------------
    signal_columns = [
        # "RMSStdDev_ms", "RRPeakCoverage", "SDNN_ms", "RR0", 
        # "sleepId", "level", 
        "bpm", "StepCount"
    ]
    scale = "mean"
    window_size = 15    # minutes
    label_type = None

    subject_ids, dates, data = load_tiles_open(
        signal_columns=signal_columns,
        scale=scale, window_size=window_size, debug=debug
    )
    labels = [-1 for _ in range(len(subject_ids))]

    # Split into train and validation sets, 80/20
    unique_subjects = list(set(subject_ids))
    train_subjects = random.sample(unique_subjects, int(len(unique_subjects)*0.8))
    train_subject_ids, train_data, train_labels = list(), list(), list()
    val_subject_ids, val_data, val_labels = list(), list(), list()
    for i in range(len(subject_ids)):
        if subject_ids[i] in train_subjects:
            train_subject_ids.append(subject_ids[i])
            train_data.append(data[i])
            train_labels.append(labels[i])
        else:
            val_subject_ids.append(subject_ids[i])
            val_data.append(data[i])
            val_labels.append(labels[i])

    tiles_train = TilesDataset(train_subject_ids, train_data, train_labels)
    tiles_val = TilesDataset(val_subject_ids, val_data, val_labels)

    train_dataloader = DataLoader(tiles_train, batch_size=batch_size, num_workers=0, shuffle=True, generator=torch.Generator(device=device))
    val_dataloader = DataLoader(tiles_val, batch_size=batch_size, num_workers=0, shuffle=False)

    num_train_samples = len(tiles_train)
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
        lr=lr,
        weight_decay=1e-4,
        betas=(0.9, 0.95)
    )

    # Read trainable params
    model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    params = sum([np.prod(p.size()) for p in model_parameters])
    logging.info(f"Trainable params: {params/(1e6):.2f} M")

    scheduler = None

    trainer = Trainer(
        n_epochs=epochs, 
        checkpoint_dir=CHECKPOINT_DIR, checkpoint_prefix=CHECKPOINT_PREFIX, 
        model_save_folder=MODEL_SAVE_FOLDER,
        mode=mode, reconstruction=reconstruction, use_wandb=use_wandb
    )

    model = trainer.train_recon(
        model, train_dataloader, val_dataloader, criterion=nn.MSELoss(),
        optimizer=optimizer, scheduler=scheduler, mask_ratio=mask_ratio,
        resume_checkpoint=None, device=device, debug=debug
    )

    if not debug:
        test_subject_ids, dates, test_data = load_tiles_holdout(
            signal_columns=signal_columns,
            scale=scale, window_size=window_size, debug=debug
        )
        test_labels = [-1 for _ in range(len(test_subject_ids))]
        tiles_test = TilesDataset(test_subject_ids, test_data, test_labels)
        test_dataloader = DataLoader(tiles_test, batch_size=batch_size, num_workers=0, shuffle=True, generator=torch.Generator(device=device))

        test_loss = trainer.validate_recon(
            model, test_dataloader, criterion=nn.MSELoss(),
            mask_ratio=mask_ratio, split="test",
            device=device
        )

    # Save model
    if not debug: save_model(model)

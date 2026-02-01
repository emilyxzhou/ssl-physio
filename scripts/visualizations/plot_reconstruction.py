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

import datetime
import json
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns
import torch

from mamba_mae import MambaMAE
from s4_mae import S4MAE
from tiles_dataloader import get_data_from_splits, get_embeddings_from_file, TilesDataset, generate_binary_labels, generate_continuous_labels_day


MODELS_BASE_PATH = f"{USER_ROOT}/ssl-physio/models/reconstruction"


def load_model(checkpoint_path, config_path, model_type, mask_ratio=None, device="cuda:1"):
    # Read arguments -----------------------------------------------------------------------------------------------
    config = json.load(open(config_path, "r"))
    model_params = config["model_params"]
    if model_params["dec_hidden_dims"] is not None: model_params["d_model"] = model_params["dec_hidden_dims"][0]
    model_params["mask_ratio"] = 0.0

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if model_type == "s4":
        model = S4MAE(
            **model_params,
            classification=False,
            device=device
        ).to(device)
    elif model_type == "mamba":
        model = MambaMAE(
            **model_params,
            classification=False,
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
    """Freeze all model weights for inference."""
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model


def plot_recon(data, mask, reconstruction, model_type, mask_pct):
    base_time = datetime.datetime(2024, 1, 1)
    time_axis = [base_time + datetime.timedelta(minutes=i) for i in range(1440)]

    # Initialize stacked subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    labels = ['Step Count', 'Heart Rate']
    colors = ["#3b8dc8", "#d67327"]

    for i, ax in enumerate(axes):
        # 1. Plot Ground Truth and Reconstruction
        ax.plot(time_axis, data[i], color='black', alpha=0.3, label='Original', linewidth=1)
        ax.plot(time_axis, reconstruction[i], color=colors[i], label='Reconstructed', linewidth=1.5)
        
        # 2. Highlight Masked Regions
        masked_indices = np.where(mask[i] == 1)[0]
        if len(masked_indices) > 0:
            change_points = np.where(np.diff(masked_indices) > 1)[0]
            starts = np.insert(change_points + 1, 0, 0)
            ends = np.append(change_points, len(masked_indices) - 1)
            
            for s, e in zip(starts, ends):
                ax.axvspan(time_axis[masked_indices[s]], time_axis[masked_indices[e]], 
                        color='gray', alpha=0.15, label='Masked' if s == 1 else "")

        # 3. Apply Specific Formatting
        ax.set_ylabel(labels[i], fontsize=10, fontweight='bold')
        
        # Remove vertical ticks and labels
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.tick_params(axis='y', left=False)
        
        # Horizontal axis formatting (only needed on the bottom plot due to sharex=True)
        if i == 1:
            ax.set_xlabel('Time (HH:MM)')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        
        ax.legend(loc='upper right', frameon=True, fontsize='small')
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    plt.suptitle(f'Masked Autoencoder Reconstruction ({model_type.capitalize()}, {mask_pct}% masking)', fontsize=14)

    plt.tight_layout()
    save_path = f"/home/emilyzho/ssl-physio/plots/reconstruction/recon_{model_type}_{mask_pct}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)


if __name__ == "__main__":
    # Find device
    device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.set_default_device("cuda:1")
        print("\Default device set to CUDA.")
    else:
        print("\nCUDA not available, using CPU.")
    torch.set_default_dtype(torch.float)

    method = "raw"
    binary_labels = ["age", "shift", "anxiety", "stress"]
    model_types = ["s4", "mamba"]
    mask_pcts = [10, 30, 50, 70]

    # Load data, raw
    print("Loading minute-level data")
    subject_ids, dates, data = get_data_from_splits()
    labels = generate_binary_labels(subject_ids, dates, label_type=binary_labels[0])
    data_df = pd.DataFrame({
        'ID': subject_ids,
        'Date': dates,
        'Data': data
    })
    test_subject = random.choice(subject_ids)
    test_date = random.choice(
        data_df.loc[(data_df['ID'] == test_subject), 'Date'].tolist()
    )

    print(f"Randomly selected subject and date: {test_subject}, {test_date}")

    for model_type in model_types:
        for mask_pct in mask_pcts:
            print("="*50)
            print(f"{model_type} {mask_pct}% masking")
            print("="*50)

            config_path = f"{USER_ROOT}/ssl-physio/config/{model_type}_config.json"
            checkpoint_path = os.path.join(MODELS_BASE_PATH, f"{model_type}-mae_{mask_pct}.pt")
            mask_ratio = mask_pct / 100.0

            # Load model
            print(f"\nLoading model from: {checkpoint_path}")
            model = load_model(
                checkpoint_path, config_path, model_type, mask_ratio=mask_ratio, device=device
            )
            model = freeze_weights(model)
            model.eval()

            test_sample = data_df.loc[(data_df['ID'] == test_subject) & (data_df['Date'] == test_date), 'Data'].iloc[0]
            test_sample = test_sample.T[np.newaxis, :]
            test_sample = torch.from_numpy(test_sample).to(device).float()

            out, target, mask = model(test_sample, mask_ratio=mask_ratio)
            mask = mask.to(int)

            conv_output = model.encoder(test_sample)  # (batch, 2, 1440) with Identity encoder
            seq_output = model.seq_model(conv_output.transpose(-1, -2))  # (batch, 1440, 128)
            seq_output = seq_output.detach().cpu().numpy()[0]

            # Get masked positions
            x_recon_masked = (out * (1-mask)).detach().cpu().numpy()
            x_masked = (target * (1-mask)).detach().cpu().numpy()

            test_sample = test_sample.detach().cpu().numpy()[0]
            mask = mask.detach().cpu().numpy()[0]
            recon = out.detach().cpu().numpy()[0]

            plot_recon(test_sample, mask, recon, model_type, mask_pct)

            
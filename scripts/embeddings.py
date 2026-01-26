import os
import sys
from pathlib import Path
USER_ROOT = str(Path(__file__).resolve().parents[2])
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
physio_data_path = os.path.join(
    USER_ROOT, "physio-data", "src"
)
sys.path.append(physio_data_path)

import argparse
import datetime
import json
import logging
import numpy as np
import pprint
import torch
import yaml

from collections import Counter
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm

from data_reader import get_data_for_subject_list
from mamba_mae import MambaMAE
from preprocessing import apply_moving_average, impute_missing
from s4_mae import S4MAE
from tiles_dataloader import get_pretrain_eval_dataloaders

# Define logging console
logging.basicConfig(
    format="%(asctime)s %(levelname)-3s ==> %(message)s", 
    level=logging.INFO, 
    datefmt="%Y-%m-%d %H:%M:%S"
)

os.environ["S4_FAST_CAUCHY"] = "0"
os.environ["S4_FAST_VAND"] = "0"
os.environ["S4_BACKEND"] = "keops"


MODELS_BASE_PATH = f"{USER_ROOT}/ssl-physio/models/reconstruction"

SAVE_BASE_DIR = "/data1/emilyzho/tiles-2018-processed/tiles-test/embeddings"

def load_model(checkpoint_path, config_path, model_type, mask_ratio=None):
    # Read arguments -----------------------------------------------------------------------------------------------
    config = json.load(open(config_path, "r"))
    model_params = config["model_params"]
    if model_params["dec_hidden_dims"] is not None: model_params["d_model"] = model_params["dec_hidden_dims"][0]
    if mask_ratio is not None: model_params["mask_ratio"] = mask_ratio

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if model_type == "s4":
        model = S4MAE(
            **model_params,
            classification=False
        ).to(device)
    elif model_type == "mamba":
        model = MambaMAE(
            **model_params,
            classification=False
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


def extract_embeddings(model, data_list, device, batch_size=32):
    """
    Extract embeddings from the encoder.
    
    Args:
        model: Trained S4MAE/Mamba model (with weights loaded, in eval mode)
        data_list: List of numpy arrays, each shape (1440, 2)
        device: torch device
        batch_size: Number of samples to process at once
    
    Returns:
        embeddings: numpy array of shape (num_samples, 128)
    
    Dimension trace:
        Input per day: (1440, 2) - 1440 minutes, 2 features (bpm, StepCount)
        After transpose + batch: (batch, 2, 1440)
        After Conv Encoder (Identity): (batch, 2, 1440)
        After S4 Model: (batch, 1440, 128)
        After mean pooling: (batch, 128) <- Final embedding
    """
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        # Process in batches for efficiency
        for i in tqdm(range(0, len(data_list), batch_size), desc="Extracting embeddings"):
            batch_data = data_list[i:i+batch_size]
            
            # Stack and transpose: list of (1440, 2) -> (batch, 2, 1440)
            batch_tensor = torch.stack([
                torch.tensor(sample, dtype=torch.float).T 
                for sample in batch_data
            ]).to(device)
            
            # Forward through encoder and S4 (but NOT the decoder)
            # No masking for embedding extraction
            conv_output = model.encoder(batch_tensor)  # (batch, 2, 1440) with Identity encoder
            seq_output = model.seq_model(conv_output.transpose(-1, -2))  # (batch, 1440, 128)
            
            # Mean pool over sequence length to get single embedding per day
            embedding = seq_output.mean(dim=1)  # (batch, 128)
            embeddings.append(embedding.cpu().numpy())
    
    return np.vstack(embeddings)  # (num_samples, 128)


def report_data_statistics(subject_ids, dates, data):
    """Report statistics about the loaded data."""
    print("\n" + "="*60)
    print("DATA STATISTICS REPORT")
    print("="*60)
    
    unique_subjects = list(set(subject_ids))
    print(f"\nTotal samples (subject-days): {len(data)}")
    print(f"Unique subjects: {len(unique_subjects)}")
    
    # Days per subject
    days_per_subject = Counter(subject_ids)
    days_counts = list(days_per_subject.values())
    print(f"\nDays per subject:")
    print(f"  Min: {min(days_counts)}")
    print(f"  Max: {max(days_counts)}")
    print(f"  Mean: {np.mean(days_counts):.1f}")
    print(f"  Median: {np.median(days_counts):.1f}")
    
    # Data shape verification
    print(f"\nData shape verification:")
    shapes = [d.shape for d in data]
    unique_shapes = list(set(shapes))
    print(f"  Unique shapes found: {unique_shapes}")
    if len(unique_shapes) == 1:
        print(f"  All samples have shape: {unique_shapes[0]}")
    else:
        print(f"  WARNING: Inconsistent shapes detected!")
        for shape in unique_shapes:
            count = shapes.count(shape)
            print(f"    {shape}: {count} samples")
    
    # Date range
    min_date = min(dates)
    max_date = max(dates)
    print(f"\nDate range: {min_date} to {max_date}")
    
    # Check for NaN values
    nan_counts = [np.sum(np.isnan(d)) for d in data]
    print(f"\nNaN values per sample:")
    print(f"  Samples with any NaN: {sum(1 for c in nan_counts if c > 0)}")
    print(f"  Max NaN count in a sample: {max(nan_counts)}")
    
    print("="*60 + "\n")
    
    return {
        "total_samples": len(data),
        "unique_subjects": len(unique_subjects),
        "days_per_subject": days_per_subject,
        "min_days": min(days_counts),
        "max_days": max(days_counts),
        "mean_days": np.mean(days_counts),
    }


def save_embeddings(embeddings, subject_ids, dates, save_dir, mask_pct):
    """Save embeddings, index, and readme to specified directory."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save embeddings as single .npy file
    embeddings_path = os.path.join(save_dir, "embeddings.npy")
    np.save(embeddings_path, embeddings)
    print(f"  Saved embeddings to: {embeddings_path}")
    
    # Create index with metadata
    index = []
    for i, (subject_id, date) in enumerate(zip(subject_ids, dates)):
        index.append({
            "key": f"{subject_id}_{date}",
            "row": i,
            "user": subject_id,
            "date": str(date)
        })
    
    # Group by user and assign user_day (1-indexed day per user)
    from collections import defaultdict
    user_entries = defaultdict(list)
    for entry in index:
        user_entries[entry["user"]].append(entry)
    
    # Sort each user's entries by date and assign user_day
    for user, entries in user_entries.items():
        entries.sort(key=lambda x: x["date"])
        for day_num, entry in enumerate(entries, start=1):
            entry["user_day"] = day_num
    
    # Rebuild index sorted by user (major) and user_day (minor)
    sorted_index = []
    for user in sorted(user_entries.keys()):
        sorted_index.extend(user_entries[user])
    
    index_path = os.path.join(save_dir, "index.json")
    with open(index_path, "w") as f:
        json.dump(sorted_index, f, indent=4)
    print(f"  Saved index to: {index_path}")
    
    # Create readme.txt
    readme_content = f"""TILES-2018 Holdout Embeddings (masking_{mask_pct})
============================================

128-dimensional embeddings from S4-MAE pretrained on TILES-2018 open dataset.
Model trained with {mask_pct}% masking ratio.
Each row = one user-day (heart rate + step count, 1440 minutes).

Files:
  embeddings.npy  - (N, 128) float64 array
  index.json      - metadata for each row, sorted by user then user_day

Index fields:
  key      - unique identifier "{user}_{date}"
  row      - numpy row index
  user     - subject ID
  date     - date string
  user_day - 1-indexed day number per user

Usage:
  import json
  import numpy as np

  embeddings = np.load("embeddings.npy")
  with open("index.json") as f:
      index = json.load(f)

  # Get embedding for specific user-day
  entry = index[0]
  emb = embeddings[entry["row"]]  # shape (128,)
"""
    
    readme_path = os.path.join(save_dir, "readme.txt")
    with open(readme_path, "w") as f:
        f.write(readme_content)
    print(f"  Saved readme to: {readme_path}")


if __name__ == "__main__":
    # Read arguments -----------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Generate embeddings from MAE models.")
    parser.add_argument("--debug", "-d", action="store_true", default=False,
                        help="If set, only loads 5 subjects for testing")
    parser.add_argument("--batch_size", "-b", type=int, default=32,
                        help="Batch size for embedding extraction")
    parser.add_argument("--mask", "-m", type=int, choices=[10, 30, 50, 70], default=None,
                        help="Run only for specific masking ratio (10, 30, 50, or 70). Default: run all.")
    parser.add_argument("--model_type", "-t", type=str, choices=["s4", "mamba"], default=None,
                        help="Run only for specific encoder type (s4 or mamba). Default: run all.")
    args = parser.parse_args()
    debug = args.debug
    batch_size = args.batch_size

    # Determine which models to run
    if args.mask is not None:
        mask_ratios = [args.mask]
    else:
        mask_ratios = [10, 30, 50, 70]

    if args.model_type is not None:
        model_types = [args.model_type]
    else:
        model_types = ["s4", "mamba"]

    print(f"\nWill generate embeddings for masking ratios: {mask_ratios} and encoders: {model_types}")

    # Find device
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.set_default_device("cuda:0")
        print("\nDefault device set to CUDA.")
    else:
        print("\nCUDA not available, using CPU.")
    torch.set_default_dtype(torch.float)

    # Loading data (only once, reused for all models) -------------------------------------------------------------
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    signal_columns = ["bpm", "StepCount"]
    scale = "mean"
    window_size = 15    # minutes

    splits = json.load(open('/data1/emilyzho/tiles-2018-processed/subject_splits.json', "r"))
    subject_ids = []
    dates = []
    data = []

    unique_subjects = splits["test"].keys()
    dfs = get_data_for_subject_list(unique_subjects)
    for data_df in dfs: 
        id = data_df["ID"].iloc[0]
        date = data_df["Date"].iloc[0]

        try: df = data_df[signal_columns]
        except Exception: continue

        for col in signal_columns:
            df.loc[:, col] = df[col].astype(float)

        # Filtering out invalid days
        nan_count = np.sum(np.isnan(np.array(df["bpm"], dtype=float)))
        if nan_count/1440 > 0.2:
            continue

        # Get indices of NaN in HR array to remove from step count array
        # nan_hr_rows = df[df["bpm"].isna()].index
        # df.loc[nan_hr_rows, "StepCount"] = np.nan

        # Perform imputation: linear interpolation on heart rate, step count
        for col in signal_columns:
            df.loc[:, col] = impute_missing(df, col, method="linear")
            df.loc[:, col] = df[col].bfill()
        df = df.bfill()

        # Apply moving averagez
        if window_size > 0:
            for col in signal_columns:
                df.loc[:, col] = apply_moving_average(df, col, window_size=window_size)

        # Scale data according to `scale` parameter
        if scale == "mean":
            scaler = StandardScaler()
            for col in signal_columns:
                df.loc[:, col] = scaler.fit_transform(df[[col]])
        elif scale == "median":
            for col in signal_columns:
                med = df[col].median()
                q1 = np.nanpercentile(df[col], 25)
                q3 = np.nanpercentile(df[col], 75)
                iqr = q3 - q1

                df.loc[:, col] = (df[col] - med) / (iqr + 1e-5)

        df = df.to_numpy()

        subject_ids.append(id)
        dates.append(date)
        data.append(df)

    # Report data statistics
    stats = report_data_statistics(subject_ids, dates, data)

    # Process each model ------------------------------------------------------------------------------------------
    for model_type in model_types:
        for mask_pct in mask_ratios:
            config_path = f"{USER_ROOT}/ssl-physio/config/{model_type}_config.json"
            checkpoint_path = os.path.join(MODELS_BASE_PATH, f"{model_type}-mae_{mask_pct}.pt")
            mask_ratio = mask_pct / 100.0
            
            print("\n" + "="*60)
            print(f"PROCESSING: masking_{mask_pct} ({mask_pct}% masking)")
            print("="*60)
            
            # Load model
            print(f"\nLoading model from: {checkpoint_path}")
            model = load_model(
                checkpoint_path, config_path, model_type, mask_ratio=mask_ratio
            )
            model = freeze_weights(model)
            print("Model loaded and frozen.")

            # Extract embeddings
            print("\nExtracting embeddings...")
            embeddings = extract_embeddings(model, data, device, batch_size=batch_size)
            
            print(f"\nEmbeddings extracted:")
            print(f"  Shape: {embeddings.shape}")
            print(f"  Memory: {embeddings.nbytes / 1024 / 1024:.2f} MB")

            # Save to subfolder
            save_dir = os.path.join(SAVE_BASE_DIR, model_type, f"masking_{mask_pct}")
            print(f"\nSaving to: {save_dir}")
            save_embeddings(embeddings, subject_ids, dates, save_dir, mask_pct)
            
            # Free memory
            del model
            torch.cuda.empty_cache()

        # Final summary
        print("\n" + "="*60)
        print("COMPLETE")
        print("="*60)
        print(f"Total samples: {len(data)}")
        print(f"Unique subjects: {stats['unique_subjects']}")
        print(f"Embeddings saved to:")
        for mask_pct in mask_ratios:
            print(f"  - {SAVE_BASE_DIR}/{model_type}/masking_{mask_pct}/")
        print("="*60)

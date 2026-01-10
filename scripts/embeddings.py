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
import json
import logging
import numpy as np
import pprint
import torch
import yaml

from collections import Counter
from tqdm import tqdm

from tiles_dataloader import load_tiles_holdout
from s4_mae import S4MAE

# Define logging console
logging.basicConfig(
    format="%(asctime)s %(levelname)-3s ==> %(message)s", 
    level=logging.INFO, 
    datefmt="%Y-%m-%d %H:%M:%S"
)

os.environ["S4_FAST_CAUCHY"] = "0"
os.environ["S4_FAST_VAND"] = "0"
os.environ["S4_BACKEND"] = "keops"

# Override the data path for tiles-holdout
import constants
constants.TILES_HOLDOUT_FITBIT_BASE_FOLDER = "/data1/mjma/tiles-holdout/fitbit"
constants.TILES_HOLDOUT_LABELS_DEMOG = "/data1/mjma/tiles-holdout/labels/demographics_1.csv"
constants.TILES_HOLDOUT_LABELS_ANXIETY = "/data1/mjma/tiles-holdout/labels/anxiety.csv"
constants.TILES_HOLDOUT_LABELS_SHIFT = "/data1/mjma/tiles-holdout/labels/shift.csv"
constants.TILES_HOLDOUT_LABELS_STRESSD = "/data1/mjma/tiles-holdout/labels/stressd.csv"

# Model configurations: (mask_ratio_percent, checkpoint_path)
MODELS = {
    10: f"{USER_ROOT}/ssl-physio/models/reconstruction/s4-mae_2025-12-22_11:15:39.pt",
    30: f"{USER_ROOT}/ssl-physio/models/reconstruction/s4-mae_2025-12-23_06:39:27.pt",
    50: f"{USER_ROOT}/ssl-physio/models/reconstruction/s4-mae_2026-01-04_21:48:55.pt",
}

SAVE_BASE_DIR = "/data1/mjma/tiles-2018-processed/tiles-holdout/embeddings"


def load_model(checkpoint_path, mask_ratio, device, d_input, d_output, enc_hidden_dims, dec_hidden_dims, n_layers_s4, verbose=False):
    """Load the S4MAE model from checkpoint."""
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
    """Freeze all model weights for inference."""
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model


def extract_embeddings(model, data_list, device, batch_size=32):
    """
    Extract embeddings from the S4MAE encoder.
    
    Args:
        model: Trained S4MAE model (with weights loaded, in eval mode)
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
                torch.tensor(sample, dtype=torch.float64).T 
                for sample in batch_data
            ]).to(device)
            
            # Forward through encoder and S4 (but NOT the decoder)
            # No masking for embedding extraction
            conv_output = model.encoder(batch_tensor)  # (batch, 2, 1440) with Identity encoder
            s4_output = model.s4_model(conv_output.transpose(-1, -2))  # (batch, 1440, 128)
            
            # Mean pool over sequence length to get single embedding per day
            embedding = s4_output.mean(dim=1)  # (batch, 128)
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
    
    # Create index.json with metadata
    index = []
    for i, (subject_id, date) in enumerate(zip(subject_ids, dates)):
        index.append({
            "key": f"{subject_id}_{date}",
            "row": i,
            "user": subject_id,
            "date": str(date)
        })
    
    index_path = os.path.join(save_dir, "index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=4)
    print(f"  Saved index to: {index_path}")
    
    # Create readme.txt
    readme_content = f"""TILES-2018 Holdout Embeddings (masking_{mask_pct})
============================================

128-dimensional embeddings from S4-MAE pretrained on TILES-2018 open dataset.
Model trained with {mask_pct}% masking ratio.
Each row = one user-day (heart rate + step count, 1440 minutes).

Files:
  embeddings.npy  - (N, 128) float64 array
  index.json      - metadata for each row

Usage:
  import json
  import numpy as np

  embeddings = np.load("embeddings.npy")
  with open("index.json") as f:
      index = json.load(f)

  # Get embedding for specific user-day
  row = index[0]["row"]
  user = index[0]["user"]
  date = index[0]["date"]
  emb = embeddings[row]  # shape (128,)
"""
    
    readme_path = os.path.join(save_dir, "readme.txt")
    with open(readme_path, "w") as f:
        f.write(readme_content)
    print(f"  Saved readme to: {readme_path}")


if __name__ == "__main__":
    # Read arguments -----------------------------------------------------------------------------------------------
    with open(f"{USER_ROOT}/ssl-physio/scripts/params.yaml", "r") as file:
        params = yaml.safe_load(file)
        d_input = params["d_input"]
        d_output = params["d_output"]
        enc_hidden_dims = params["enc_hidden_dims"]
        dec_hidden_dims = params["dec_hidden_dims"]
        d_model = params["d_model"]
        n_layers_s4 = params["n_layers_s4"]
    if dec_hidden_dims is not None: 
        d_model = dec_hidden_dims[0]

    parser = argparse.ArgumentParser(description="Generate embeddings from S4-MAE models.")
    parser.add_argument("--debug", "-d", action="store_true", default=False,
                        help="If set, only loads 5 subjects for testing")
    parser.add_argument("--batch_size", "-b", type=int, default=32,
                        help="Batch size for embedding extraction")
    parser.add_argument("--mask", "-m", type=int, choices=[10, 30, 50], default=None,
                        help="Run only for specific masking ratio (10, 30, or 50). Default: run all.")
    args = parser.parse_args()
    debug = args.debug
    batch_size = args.batch_size

    # Determine which models to run
    if args.mask is not None:
        models_to_run = {args.mask: MODELS[args.mask]}
    else:
        models_to_run = MODELS

    print("\nModel parameters:")
    pprint.pprint(params)
    print(f"\nWill generate embeddings for masking ratios: {list(models_to_run.keys())}")

    # Find device
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.set_default_device("cuda:0")
        print("\nDefault device set to CUDA.")
    else:
        print("\nCUDA not available, using CPU.")
    torch.set_default_dtype(torch.float64)

    # Loading data (only once, reused for all models) -------------------------------------------------------------
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    signal_columns = ["bpm", "StepCount"]
    scale = "mean"
    window_size = 15    # minutes
    
    subject_ids, dates, data = load_tiles_holdout(
        signal_columns=signal_columns,
        scale=scale, window_size=window_size, debug=debug
    )

    # Report data statistics
    stats = report_data_statistics(subject_ids, dates, data)
    print(f"Expected embedding shape: ({len(data)}, {d_output})")

    # Process each model ------------------------------------------------------------------------------------------
    for mask_pct, checkpoint_path in models_to_run.items():
        mask_ratio = mask_pct / 100.0
        
        print("\n" + "="*60)
        print(f"PROCESSING: masking_{mask_pct} ({mask_pct}% masking)")
        print("="*60)
        
        # Load model
        print(f"\nLoading model from: {checkpoint_path}")
        model = load_model(
            checkpoint_path, mask_ratio, device,
            d_input=d_input, d_output=d_output,
            enc_hidden_dims=enc_hidden_dims, dec_hidden_dims=dec_hidden_dims,
            n_layers_s4=n_layers_s4, verbose=False
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
        save_dir = os.path.join(SAVE_BASE_DIR, f"masking_{mask_pct}")
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
    for mask_pct in models_to_run.keys():
        print(f"  - {SAVE_BASE_DIR}/masking_{mask_pct}/")
    print("="*60)

import os
import sys
from pathlib import Path
USER_ROOT = str(Path(__file__).resolve().parents[2])
paths = [
    os.path.join(
        USER_ROOT, "ssl-physio", "src"
    ),
    os.path.join(
        USER_ROOT, "ssl-physio", "src", "dataloaders"
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

import copy
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import constants

from collections import Counter
from pathlib import Path
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

from tiles_dataloader import TilesDataset, get_data_from_splits, generate_binary_labels, generate_continuous_labels_day
from utils import get_kfold_loaders


def aggregate_labels():
    debug = False

    labels_df = {
        "ID": [],
        "Date": [],
        "age": [],
        "shift": [],
        "anxiety": [],
        "stress": [],
        "NumberSteps": [],
        "RestingHeartRate": [],
        "SleepMinutesAsleep": []
    }

    subject_ids, dates, data = get_data_from_splits("test")
    label_types = ['NumberSteps', 'RestingHeartRate', 'SleepMinutesAsleep']
    all_labels = generate_continuous_labels_day(subject_ids, dates, label_types=label_types)
    for i, label_type in enumerate(label_types):
        labels = all_labels[label_type]
        # subject_ids_copy = copy.deepcopy(subject_ids)
        # dates_copy = copy.deepcopy(dates)
        # nan_indices = [i for i in range(len(labels)) if np.isnan(labels[i])]
        # nan_indices.sort(reverse=True)
        # for idx in nan_indices:
        #     subject_ids_copy.pop(idx)
        #     dates_copy.pop(idx)
        #     labels.pop(idx)

        if i == 0:
            labels_df["ID"].extend(subject_ids)
            labels_df["Date"].extend(dates)
        labels_df[label_type].extend(labels)

    for label_type in ["age", "shift", "anxiety", "stress"]:
        labels = generate_binary_labels(subject_ids, dates, label_type=label_type)
        labels_df[label_type].extend(labels)
    
    labels_df = pd.DataFrame(labels_df).dropna(how="any").sort_values(by=["ID", "Date"]).reset_index(drop=True)
    labels_df = labels_df.to_numpy()

    save_path = "/data1/mjma/tiles-2018-processed/tiles-test/labels.npy"
    np.save(save_path, labels_df)


def save_embeddings(data, subject_ids, dates, save_dir):
    """Save embeddings, index, and readme to specified directory."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save embeddings as single .npy file
    data_path = os.path.join(save_dir, "data.npy")
    np.save(data_path, data)
    print(f"  Saved data to: {data_path}")
    
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
    readme_content = f"""TILES-2018 minute-level data)
============================================
Each row = one user-day (heart rate + step count, 1440 minutes).

Files:
  data.npy  - (N, 2, 1440) float64 array
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

  data = np.load("data.npy")
  with open("index.json") as f:
      index = json.load(f)

  # Get embedding for specific user-day
  entry = index[0]
  sample = data[entry["row"]]  # shape (2,1440)
"""
    
    readme_path = os.path.join(save_dir, "readme.txt")
    with open(readme_path, "w") as f:
        f.write(readme_content)
    print(f"  Saved readme to: {readme_path}")


if __name__ == "__main__":
    save_dir = '/data1/mjma/tiles-2018-processed/tiles-test/data'
    subject_ids, dates, data = get_data_from_splits()

    print(f"\nSaving to: {save_dir}")
    save_embeddings(data, subject_ids, dates, save_dir)
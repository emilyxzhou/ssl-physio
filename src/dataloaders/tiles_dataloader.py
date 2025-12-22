import os
import sys
from pathlib import Path
src_path = os.path.join(
    str(Path(__file__).resolve().parents[3]),
    "physio-data", "src"
)
sys.path.insert(0, src_path)

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import glob
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

import constants

from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from tqdm import tqdm

from data_reader import get_data
from preprocessing import apply_moving_average, impute_missing


# Define logging console
import logging
logging.basicConfig(
    format="%(asctime)s %(levelname)-3s ==> %(message)s", 
    level=logging.INFO, 
    datefmt="%Y-%m-%d %H:%M:%S"
)


def load_tiles_open(
        signal_columns,
        scale="mean", window_size=0, 
        label_type="shift",
        debug=True
    ):
    """
    Returns minute-level data from the TILES-2018 open dataset.
    
    :param scale: None (no scaling), "mean" (mean & std, day-level), "median" (med & iqr, day-level), 
        "global mean" (mean & std, across all subjects and days)
    :param window_size: Window size for moving average calculation. Ignores if 0 is passed.
    :param label_type: 
        - None
        - From EMAs: "shift", "age", "sex", "anxiety", "stress"
        - Wearable-based, next-day averages: "day RHR", "day step count", "day SDNN", 
    :param debug: If True, only loads 
    """
    data_dfs = get_data(constants.TILES_OPEN_FITBIT_BASE_FOLDER, debug=debug)
    data = list()
    dates = list()
    subject_ids = list()
    labels = list()

    if label_type is not None:
        if label_type == "age": 
            label_file = constants.TILES_OPEN_LABELS_DEMOG
            label_col = "age"
        elif label_type == "sex": 
            label_file = constants.TILES_OPEN_LABELS_DEMOG
            label_col = "gender"
        elif label_type == "anxiety": 
            label_file = constants.TILES_OPEN_LABELS_ANXIETY
            label_col = "anxiety"
        elif label_type == "stress": 
            label_file = constants.TILES_OPEN_LABELS_STRESSD
            label_col = "stressd"
        elif label_type == "shift": 
            label_file = constants.TILES_OPEN_LABELS_SHIFT
            label_col = "Primary Shift"
        label_df = pd.read_csv(label_file, index_col=0)

    for data_df in data_dfs: 
        id = data_df["ID"].iloc[0]
        date = data_df["Date"].iloc[0]

        if label_type is not None:
            label = label_df.loc[(label_df["ID"] == id) & (data_df["Date"] == date), label_col]
            if label.empty: 
                # logging.info("Empty label row, skipping sample.")
                continue
            else: label = label.values[0]
        else: label = -1

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

        # Apply moving average
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
        labels.append(label)

    return subject_ids, dates, data, labels    


def load_tiles_holdout(
        signal_columns,
        scale="mean", window_size=0, 
        label_type="shift",
        debug=True
    ):
    """
    Returns minute-level data from the TILES-2018 held-out dataset.
    
    :param scale: None (no scaling), "mean" (mean & std, day-level), "median" (med & iqr, day-level), 
        "global mean" (mean & std, across all subjects and days)
    :param window_size: Window size for moving average calculation. Ignores if 0 is passed.
    :param label_type: "shift", "age", "sex", "anxiety", "stress", None
    :param debug: If True, only loads 
    """
    data_dfs = get_data(constants.TILES_HOLDOUT_FITBIT_BASE_FOLDER, debug=debug)
    data = list()
    dates = list()
    subject_ids = list()
    labels = list()

    if label_type is not None:
        if label_type == "age": 
            label_file = constants.TILES_HOLDOUT_LABELS_DEMOG
            label_col = "age"
        elif label_type == "sex": 
            label_file = constants.TILES_HOLDOUT_LABELS_DEMOG
            label_col = "gender"
        elif label_type == "anxiety": 
            label_file = constants.TILES_HOLDOUT_LABELS_ANXIETY
            label_col = "anxiety"
        elif label_type == "stress": 
            label_file = constants.TILES_HOLDOUT_LABELS_STRESSD
            label_col = "stressd"
        elif label_type == "shift": 
            label_file = constants.TILES_HOLDOUT_LABELS_SHIFT
            label_col = "Primary Shift"
        label_df = pd.read_csv(label_file, index_col=0)

    for data_df in data_dfs: 
        id = data_df["ID"].iloc[0]
        date = data_df["Date"].iloc[0]

        if label_type is not None:
            label = label_df.loc[(label_df["ID"] == id) & (data_df["Date"] == date), label_col]
            if label.empty: 
                # logging.info("Empty label row, skipping sample.")
                continue
            else: label = label.values[0]
        else: label = -1

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

        # Apply moving average
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
        labels.append(label)

    return subject_ids, dates, data, labels    


class TilesDataset(Dataset):

    def __init__(
        self, subject_ids=None, data=None, labels=None
    ):

        if data is not None and labels is not None:
            assert subject_ids is not None
            assert len(subject_ids) == len(data) == len(labels)

            self.data = data
            self.subject_ids = subject_ids
            self.labels = labels

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data       = self.data[index]
        subject_id = self.subject_ids[index]
        labels     = self.labels[index]
        return data, subject_id, labels


if __name__ == "__main__":
    signal_columns = [
        # "RMSStdDev_ms", "RRPeakCoverage", "SDNN_ms", "RR0", 
        # "sleepId", "level", 
        "bpm", "StepCount"
    ]
    scale = "mean"
    window_size = 15    # minutes
    label_type = None
    debug = True

    # subject_ids, dates, data, labels = load_tiles_open(
    #     signal_columns=signal_columns,
    #     scale=scale, window_size=window_size, label_type=label_type, debug=debug
    # )

    # print(f"{len(subject_ids)}, {len(dates)}, {len(data)}, {len(labels)}")

    # print(subject_ids[0])
    # print(dates[0])
    # print(data[0])
    # print(labels[0])


    subject_ids, dates, data, labels = load_tiles_holdout(
        signal_columns=signal_columns,
        scale=scale, window_size=window_size, label_type=label_type, debug=debug
    )

    print(f"{len(subject_ids)}, {len(dates)}, {len(data)}, {len(labels)}")

    print(subject_ids[0])
    print(dates[0])
    print(data[0])
    print(labels[0])
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
warnings.filterwarnings(action="ignore", category=RuntimeWarning, message="invalid value encountered in divide")
warnings.filterwarnings(action="ignore", category=RuntimeWarning, message="Mean of empty slice")

import datetime
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

from data_reader import get_data, get_data_for_subject, get_data_daily
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
        debug=True
    ):
    """
    Returns minute-level data from the TILES-2018 open dataset.
    
    :param scale: None (no scaling), "mean" (mean & std, day-level), "median" (med & iqr, day-level), 
        "global mean" (mean & std, across all subjects and days)
    :param window_size: Window size for moving average calculation. Ignores if 0 is passed.
    :param debug: If True, only loads five subjects
    """
    data_dfs = get_data(constants.TILES_OPEN_FITBIT_BASE_FOLDER, debug=debug)
    data = list()
    dates = list()
    subject_ids = list()

    for data_df in data_dfs: 
        id = data_df["ID"].iloc[0]
        date = data_df["Date"].iloc[0]

        try: df = data_df[signal_columns]
        except Exception: continue

        for col in signal_columns:
            df.loc[:, col] = df[col].astype(float)

        # Filtering out invalid days
        num_invalid = 0
        nan_count = np.sum(np.isnan(np.array(df["bpm"], dtype=float)))
        if nan_count/1440 > 0.2:
            num_invalid += 1
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
    print(f"{num_invalid} days discarded.")
    return subject_ids, dates, data    


def load_tiles_holdout(
        signal_columns,
        scale="mean", window_size=0, 
        debug=True
    ):
    """
    Returns minute-level data from the TILES-2018 held-out dataset.
    
    :param scale: None (no scaling), "mean" (mean & std, day-level), "median" (med & iqr, day-level), 
        "global mean" (mean & std, across all subjects and days)
    :param window_size: Window size for moving average calculation. Ignores if 0 is passed.
    :param debug: If True, only loads five subjects
    """
    data_dfs = get_data(constants.TILES_HOLDOUT_FITBIT_BASE_FOLDER, debug=debug)
    data = list()
    dates = list()
    subject_ids = list()

    for data_df in data_dfs: 
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

    return subject_ids, dates, data    


def generate_binary_labels(subject_ids, dates, version, label_type):
    """
    Generate corresponding binary labels from demographics and EMAs for the given lists of subject IDs and dates.
    Age: > 40 --> 1, <= 40 --> 0
    Shift: 
    
    :param subject_ids: list of subject IDs obtained from `load_tiles_open` or `load_tiles_holdout`
    :param dates: list of dates obtained from `load_tiles_open` or `load_tiles_holdout`
    :param version: "open", "holdout"
    :param label_type: "shift", "age", "sex", "anxiety", "stress", None
    """
    labels = list()
    if label_type == "age": 
        if version == "open": label_file = constants.TILES_OPEN_LABELS_DEMOG
        else: label_file = constants.TILES_HOLDOUT_LABELS_DEMOG
        label_col = "age"
    elif label_type == "sex": 
        if version == "open": label_file = constants.TILES_OPEN_LABELS_DEMOG
        else: label_file = constants.TILES_HOLDOUT_LABELS_DEMOG
        label_col = "gender"
    elif label_type == "anxiety": 
        if version == "open": label_file = constants.TILES_OPEN_LABELS_ANXIETY
        else: label_file = constants.TILES_HOLDOUT_LABELS_ANXIETY
        label_col = "anxiety"
    elif label_type == "stress": 
        if version == "open": label_file = constants.TILES_OPEN_LABELS_STRESSD
        else: label_file = constants.TILES_HOLDOUT_LABELS_STRESSD
        label_col = "stressd"
    elif label_type == "shift": 
        if version == "open": label_file = constants.TILES_OPEN_LABELS_SHIFT
        else: label_file = constants.TILES_HOLDOUT_LABELS_SHIFT
        label_col = "Primary Shift"
    label_df = pd.read_csv(label_file, index_col=0)
    
    if "Date" in label_df.columns:
        label_df["Date"] = label_df["Date"].apply(lambda d: datetime.datetime.strptime(d, "%Y-%m-%d").date())

    subject_labels = {subject_id: [] for subject_id in list(set(subject_ids))}
    for i in range(len(subject_ids)):
        subject_id = subject_ids[i]
        date = dates[i]
        
        if label_type in ["anxiety", "stress"]:
            label = label_df.loc[(label_df["ID"] == subject_id) & (label_df["Date"] == date), label_col]
        else:
            label = label_df.loc[label_df["ID"] == subject_id, label_col]
        if label.empty: 
            # logging.info("Empty label row, skipping sample.")
            label = np.nan
        else: label = label.values[0]

        labels.append(label)
        subject_labels[subject_id].append(label)

    # Label threshold if applicable
    if label_type in ["anxiety", "stress"]:
        for subject_id in subject_labels.keys():
            average_label = np.nanmean(subject_labels[subject_id])
            subject_labels[subject_id] = average_label

    for i in range(len(subject_ids)):
        subject_id = subject_ids[i]
        label = labels[i]

        if label_type == "age":
            if label > 40: labels[i] = 1
            else: labels[i] = 0
        elif label_type == "sex":
            if label == 2: labels[i] = 1
            else: labels[i] = 0
        elif label_type == "shift":
            if label == "Day shift": labels[i] = 0
            else: labels[i] = 1
        elif label_type in ["anxiety", "stress"]:
            if label > subject_labels[subject_id]: labels[i] = 1
            else: labels[i] = 0

    return labels


def generate_continuous_labels_day(subject_ids, dates, version, label_type, debug=False):
    """
    Generate corresponding continuous labels from demographics and wearable data for the given lists of subject IDs and dates.
    
    :param subject_ids: list of subject IDs obtained from `load_tiles_open` or `load_tiles_holdout`
    :param dates: list of dates obtained from `load_tiles_open` or `load_tiles_holdout`
    :param version: "open", "holdout"
    :param label_type: "RMSStdDev_ms", "RRPeakCoverage", "SDNN_ms", "RR0 ", "bpm", "level"
        - The following are obtained from the daily summary files located at '/data1/tiantiaf/tiles-opendata/tiles-phase1-opendataset-holdout/fitbit/daily-summary/{subject_ID}'
        - "NumberSteps", "RestingHeartRate", "SleepMinutesAsleep"
    """
    labels = list()
    unique_subject_ids = list(set(subject_ids))

    if version == "open": base_path = constants.TILES_OPEN_FITBIT_BASE_FOLDER
    else: base_path = constants.TILES_HOLDOUT_FITBIT_BASE_FOLDER
    
    if label_type in ["RMSStdDev_ms", "RRPeakCoverage", "SDNN_ms", "RR0 ", "bpm", "level"]:
        daily_data = get_data_daily(base_path, debug=debug)
        for i in range(len(subject_ids)):
            subject_id = subject_ids[i]
            date = dates[i]
            next_day = date + datetime.timedelta(days=1)
            if next_day in daily_data["Date"].values:
                try:
                    label = daily_data.loc[(daily_data["ID"] == subject_id ) & (daily_data["Date"] == next_day), label_type].item()
                except Exception as e: 
                    label = np.nan
            else: label = np.nan
            labels.append(label)
    else:
        summary_files = glob.glob("/data1/tiantiaf/tiles-opendata/tiles-phase1-opendataset-holdout/fitbit/daily-summary/*")
        summary_dfs = []
        for fp in summary_files:
            subject_id = fp.split("/")[-1].split(".")[0]
            df = pd.read_csv(fp)
            df["Date"] = df["Timestamp"].apply(lambda t: datetime.datetime.strptime(t, '%Y-%m-%d').date())
            df = df.sort_values(by="Date").reset_index(drop=True)
            df.insert(0, "ID", subject_id)
            summary_dfs.append(df)
        summary_dfs = pd.concat(summary_dfs, axis=0).reset_index(drop=True)
        for i in range(len(subject_ids)):
            subject_id = subject_ids[i]
            date = dates[i]
            try:
                label = summary_dfs[(summary_dfs["ID"] == subject_id) & (summary_dfs["Date"] == date)][label_type].iloc[0]
            except Exception as e: 
                label = np.nan
            labels.append(label)

    return labels


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
        return data.astype(np.float32), subject_id, labels


if __name__ == "__main__":
    signal_columns = [
        # "RMSStdDev_ms", "RRPeakCoverage", 
        # "SDNN_ms", 
        # "RR0", 
        # "level", 
        "bpm", "StepCount"
    ]
    scale = "mean"
    window_size = 15    # minutes

    label_types = [
        # constants.Labels.AGE,
        # constants.Labels.SHIFT,
        # constants.Labels.ANXIETY,
        # constants.Labels.STRESS,
        constants.Labels.HR,
        constants.Labels.RHR,
        constants.Labels.STEPS,
        constants.Labels.SLEEP_MINS,
    ]
    debug = True

# Test TILES-2018 open dataset ----------------------------------------------------------------------------------------------------
    # subject_ids, dates, data = load_tiles_open(
    #     signal_columns=signal_columns,
    #     scale=scale, window_size=window_size, debug=debug
    # )

    # labels = generate_binary_labels(subject_ids, dates, version="open", label_type=label_type)

# Test TILES-2018 held-out dataset ----------------------------------------------------------------------------------------------------
    subject_ids, dates, data = load_tiles_holdout(
        signal_columns=signal_columns,
        scale=scale, window_size=window_size, debug=debug
    )

    # labels = generate_binary_labels(subject_ids, dates, version="holdout", label_type=label_type)
    # print(labels)

    for label_type in label_types:
        labels = generate_continuous_labels_day(subject_ids, dates, version="holdout", label_type=label_type, debug=debug)
        print(labels)

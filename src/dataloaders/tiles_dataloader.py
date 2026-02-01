import os
import sys
from pathlib import Path
physio_src_path = os.path.join(
    str(Path(__file__).resolve().parents[3]),
    "physio-data", "src"
)
ssl_src_path = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, physio_src_path)
sys.path.insert(0, ssl_src_path)

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", category=RuntimeWarning, message="invalid value encountered in divide")
warnings.filterwarnings(action="ignore", category=RuntimeWarning, message="Mean of empty slice")

import datetime
import glob
import json
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

import constants

from scipy.signal import savgol_filter
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from data_reader import get_data, get_data_for_subject, get_data_for_subject_list, get_data_daily
from preprocessing import apply_moving_average, impute_missing
from utils import normalize_list, stratified_group_split


# Define logging console
import logging
logging.basicConfig(
    format="%(message)s", 
    level=logging.INFO, 
    # datefmt="%Y-%m-%d %H:%M:%S"
)


class TilesDataset(Dataset):

    def __init__(
        self, subject_ids=None, data=None, labels=None
    ):

        if data is not None and labels is not None:
            assert subject_ids is not None
            assert len(subject_ids) == len(data) == len(labels)

            self._data = data
            self._subject_ids = subject_ids
            self._labels = labels

    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, index):
        data       = self._data[index]
        subject_id = self._subject_ids[index]
        labels     = self._labels[index]
        return data.astype(np.float32), subject_id, labels
    
    @property
    def subject_ids(self):
        return self._subject_ids
    
    @subject_ids.setter
    def subject_ids(self, value):
        self._subject_ids = value
    
    @property
    def labels(self):
        return self._labels
    
    @labels.setter
    def labels(self, value):
        self._labels = value


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
    # print(f"{num_invalid} days discarded.")
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


def get_pretrain_eval_dataloaders(
        batch_size=32, 
        device="cuda", 
        random_seed=42
    ):
    """
    Docstring for get_pretrain_eval_dataloaders
    
    :param signal_columns: Description
    :param label_type: None (pre-training)
        - Classification: "shift", "age", "sex", "anxiety", "stress"
        - Regression: "RMSStdDev_ms", "RRPeakCoverage", "SDNN_ms", "RR0 ", "bpm", "level"
            - The following are obtained from the daily summary files located at '/data1/tiantiaf/tiles-opendata/tiles-phase1-opendataset-holdout/fitbit/daily-summary/{subject_ID}'
            - "NumberSteps", "RestingHeartRate", "SleepMinutesAsleep"
    :param scale: Description
    :param window_size: Description
    :param test_size: Description
    :param batch_size: Description
    :param train_test_split: float; percentage of total pre-training data to use for pre-training validation
    :param device: Description
    :param debug: Description
    :param random_seed: Description
    """
    subject_ids, dates, data = get_data_from_splits("pretrain")
    train_dataset = pd.DataFrame({
        'ID': subject_ids,
        'Date': dates,
        'Data': data
    })

    gss = GroupShuffleSplit(n_splits=1, test_size=1/11, random_state=random_seed)
    train_indices, test_indices = next(gss.split(train_dataset, groups=train_dataset['ID']))
    train_df = train_dataset.iloc[train_indices]
    test_df = train_dataset.iloc[test_indices]

    pretrain_subject_ids = train_df['ID'].tolist()
    pretrain_dates = train_df['Date'].tolist()
    pretrain_data = train_df['Data'].tolist()
    pretrain_labels = [-1 for _ in range(len(pretrain_subject_ids))]

    val_subject_ids = test_df['ID'].tolist()
    val_dates = test_df['Date'].tolist()
    val_data = test_df['Data'].tolist()
    val_labels = [-1 for _ in range(len(val_subject_ids))]

    test_subject_ids, test_dates, test_data = get_data_from_splits("test")
    test_labels = [-1 for _ in range(len(test_subject_ids))]

    pretrain_dataset = TilesDataset(pretrain_subject_ids, pretrain_data, pretrain_labels)
    pretrain_dataloader = DataLoader(pretrain_dataset, batch_size=batch_size, num_workers=0, shuffle=True, generator=torch.Generator(device=device))

    val_dataset = TilesDataset(val_subject_ids, val_data, val_labels)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, shuffle=True, generator=torch.Generator(device=device))

    test_dataset = TilesDataset(test_subject_ids, test_data, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, shuffle=True, generator=torch.Generator(device=device))

    return pretrain_dataloader, val_dataloader, test_dataloader


def generate_binary_labels(subject_ids, dates, version="open", label_type="age", verbose=False):
    # NOTE: `version` is no longer used; keeping it for backwards compatibility
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
        open_file = constants.TILES_OPEN_LABELS_DEMOG
        holdout_file = constants.TILES_HOLDOUT_LABELS_DEMOG
        open_df = pd.read_csv(open_file, index_col=0)
        holdout_df = pd.read_csv(holdout_file, index_col=0)
        label_df = pd.concat([open_df, holdout_df], axis=0).reset_index(drop=True)
        label_col = "age"

    elif label_type == "sex": 
        open_file = constants.TILES_OPEN_LABELS_DEMOG
        holdout_file = constants.TILES_HOLDOUT_LABELS_DEMOG
        open_df = pd.read_csv(open_file, index_col=0)
        holdout_df = pd.read_csv(holdout_file, index_col=0)
        label_df = pd.concat([open_df, holdout_df], axis=0).reset_index(drop=True)
        label_col = "gender"

    elif label_type == "anxiety": 
        open_file = constants.TILES_OPEN_LABELS_ANXIETY
        holdout_file = constants.TILES_HOLDOUT_LABELS_ANXIETY
        open_df = pd.read_csv(open_file, index_col=0)
        holdout_df = pd.read_csv(holdout_file, index_col=0)
        label_df = pd.concat([open_df, holdout_df], axis=0).reset_index(drop=True)
        label_col = "anxiety"
        
    elif label_type == "stress": 
        open_file = constants.TILES_OPEN_LABELS_STRESSD
        holdout_file = constants.TILES_HOLDOUT_LABELS_STRESSD
        open_df = pd.read_csv(open_file, index_col=0)
        holdout_df = pd.read_csv(holdout_file, index_col=0)
        label_df = pd.concat([open_df, holdout_df], axis=0).reset_index(drop=True)
        label_col = "stressd"

    elif label_type == "shift": 
        open_file = constants.TILES_OPEN_LABELS_SHIFT
        holdout_file = constants.TILES_HOLDOUT_LABELS_SHIFT
        open_df = pd.read_csv(open_file, index_col=0)
        holdout_df = pd.read_csv(holdout_file, index_col=0)
        label_df = pd.concat([open_df, holdout_df], axis=0).reset_index(drop=True)
        label_col = "Primary Shift"

    if "Date" in label_df.columns:
        try:
            label_df["Date"] = label_df["Date"].apply(
                lambda d: datetime.datetime.strptime(d, "%Y-%m-%d").date() if type(d) is str else np.nan
            )
        except Exception as e:
            print(f"Error converting '{label_type}' date column to datetime.")
            # print(label_df["Date"].dtype)
            # print(type(label_df["Date"].iloc[0]))
            # print(label_df["Date"].iloc[0])

    label_df = label_df[label_df['ID'].isin(set(subject_ids))].reset_index(drop=True)
    # if verbose:
    #     with pd.option_context('display.max_rows', None):
    #         print(label_df[['ID', 'Date', label_col]])

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


def generate_continuous_labels_day(subject_ids, dates, version="open", label_types=["bpm"]):
    # NOTE: `version` is no longer used; keeping it for backwards compatibility
    """
    Generate corresponding continuous labels from demographics and wearable data for the given lists of subject IDs and dates.
    Uses min-max normalization.
    
    :param subject_ids: list of subject IDs obtained from `load_tiles_open` or `load_tiles_holdout`
    :param dates: list of dates obtained from `load_tiles_open` or `load_tiles_holdout`
    :param version: "open", "holdout"
    :param label_type: "RMSStdDev_ms", "RRPeakCoverage", "SDNN_ms", "RR0 ", "bpm", "level"
        - The following are obtained from the daily summary files located at '/data1/tiantiaf/tiles-opendata/tiles-phase1-opendataset-holdout/fitbit/daily-summary/{subject_ID}'
        - "NumberSteps", "RestingHeartRate", "SleepMinutesAsleep"
    """
    if type(label_types) is not list: label_types = [label_types]
    labels = {label: [] for label in label_types}

    daily_labels = list(set(label_types).intersection(("RMSStdDev_ms", "RRPeakCoverage", "SDNN_ms", "RR0 ", "bpm", "level")))
    summary_labels = list(set(label_types).intersection(("RestingHeartRate", "SleepMinutesAsleep", "NumberSteps")))

    if len(daily_labels) > 0:
        open_data = get_data_daily(constants.TILES_OPEN_FITBIT_BASE_FOLDER, debug=False)
        holdout_data = get_data_daily(constants.TILES_HOLDOUT_FITBIT_BASE_FOLDER, debug=False)
        daily_data = pd.concat([open_data, holdout_data], axis=0).reset_index(drop=True)

        for i in range(len(subject_ids)):
            subject_id = subject_ids[i]
            date = dates[i]
            next_day = date + datetime.timedelta(days=1)
            if next_day in daily_data["Date"].values:
                for label_type in daily_labels:
                    try:
                        label = daily_data.loc[(daily_data["ID"] == subject_id ) & (daily_data["Date"] == next_day), label_type].item()
                    except Exception as e: 
                        label = np.nan
                    labels[label_type].append(label)
            else: 
                for label_type in daily_labels:
                    labels[label_type].append(np.nan)
            
<<<<<<< HEAD
    else:    # "RestingHeartRate", "SleepMinutesAsleep", "NumberSteps"
        summary_files = glob.glob("/data1/tiantiaf/tiles-opendata/tiles-phase1-*/fitbit/daily-summary/*")
=======
    if len(summary_labels) > 0:
        summary_files = glob.glob("/data/tiantiaf/tiles-opendataset/tiles-phase1-*/fitbit/daily-summary/*")
>>>>>>> 26b9a0b46a2aa211bf3764c21e19968ba0bd5567
        summary_dfs = []
        for fp in summary_files:
            subject_id = fp.split("/")[-1].split(".")[0]
            try:
                df = pd.read_csv(fp)
            except Exception: 
                continue
            df["Date"] = df["Timestamp"].apply(lambda t: datetime.datetime.strptime(t, '%Y-%m-%d').date())
            df = df.sort_values(by="Date").reset_index(drop=True)
            df.insert(0, "ID", subject_id)
            summary_dfs.append(df)
        summary_dfs = pd.concat(summary_dfs, axis=0).reset_index(drop=True)

        for i in range(len(subject_ids)):
            subject_id = subject_ids[i]
            date = dates[i]
            next_day = date + datetime.timedelta(days=1)
            for label_type in summary_labels:
                if label_type in ["RestingHeartRate", "NumberSteps"]:
                    try: 
                        label = summary_dfs[(summary_dfs["ID"] == subject_id) & (summary_dfs["Date"] == next_day)][label_type].iloc[0]
                    except Exception as e: 
                        label = np.nan
                else:    # "SleepMinutesAsleep"
                    try:
                        label = summary_dfs[(summary_dfs["ID"] == subject_id) & (summary_dfs["Date"] == date)][label_type].iloc[0]
                    except Exception as e: 
                        label = np.nan
                labels[label_type].append(label)

    for label_type in labels.keys():
        labels[label_type] = normalize_list(labels[label_type])

    # Backwards compatibility
    if len(list(labels.keys())) == 1: return labels[list(labels.keys())[0]]

    return labels


def get_dataloaders(
        signal_columns, version, label_type=None,
        scale="mean", window_size=0, 
        test_size=0.2, batch_size=32,
        device="cuda", debug=True
    ):
    """
    Returns train and test DataLoaders based on test_size.
    
    :param signal_columns: Description
    :param version: "open", "holdout"
    :param label_type: None (pre-training)
        - Classification: "shift", "age", "sex", "anxiety", "stress"
        - Regression: "RMSStdDev_ms", "RRPeakCoverage", "SDNN_ms", "RR0 ", "bpm", "level"
            - The following are obtained from the daily summary files located at '/data1/tiantiaf/tiles-opendata/tiles-phase1-opendataset-holdout/fitbit/daily-summary/{subject_ID}'
            - "NumberSteps", "RestingHeartRate", "SleepMinutesAsleep"
    :param scale: Description
    :param window_size: Description
    :param test_size: Description
    :param batch_size: Description
    :param debug: True/False
    """
    if version == "open": 
        subject_ids, dates, data = load_tiles_open(
            signal_columns=signal_columns,
            scale=scale, window_size=window_size, debug=debug
        )
    else:
        subject_ids, dates, data = load_tiles_holdout(
            signal_columns=signal_columns,
            scale=scale, window_size=window_size, debug=debug
        )
    
    if label_type is None:
        labels = [-1 for _ in range(len(subject_ids))]
    elif label_type in ["shift", "age", "sex", "anxiety", "stress"]:
        labels = generate_binary_labels(subject_ids, dates, version=version, label_type=label_type)
    else:
        labels = generate_continuous_labels_day(subject_ids, dates, version=version, label_types=label_type, debug=debug)

    if test_size > 0:
        unique_subjects = list(set(subject_ids))
        train_subjects = random.sample(unique_subjects, int(len(unique_subjects) * (1-test_size)))
        train_subject_ids, train_data, train_labels = list(), list(), list()
        test_subject_ids, test_data, test_labels = list(), list(), list()
        for i in range(len(subject_ids)):
            if subject_ids[i] in train_subjects:
                train_subject_ids.append(subject_ids[i])
                train_data.append(data[i])
                train_labels.append(labels[i])
            else:
                test_subject_ids.append(subject_ids[i])
                test_data.append(data[i])
                test_labels.append(labels[i])

        tiles_train = TilesDataset(train_subject_ids, train_data, train_labels)
        tiles_test = TilesDataset(test_subject_ids, test_data, test_labels)

        train_dataloader = DataLoader(tiles_train, batch_size=batch_size, num_workers=0, shuffle=True, generator=torch.Generator(device=device))
        test_dataloader = DataLoader(tiles_test, batch_size=batch_size, num_workers=0, shuffle=False)

    else:
        tiles_train = TilesDataset(subject_ids, data, labels)
        tiles_test = None

        train_dataloader = DataLoader(tiles_train, batch_size=batch_size, num_workers=0, shuffle=True, generator=torch.Generator(device=device))
        test_dataloader = None

    return train_dataloader, test_dataloader


def get_data_from_splits(split="test"):
    signal_columns = ["bpm", "StepCount"]
    scale = "mean"
    window_size = 15    # minutes
    splits = json.load(open('/data1/emilyzho/tiles-2018-processed/subject_splits.json', 'r'))
    subject_ids = []
    dates = []
    data = []

    unique_subjects = splits[split].keys()
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

    dates = [datetime.datetime.strptime(d, "%Y-%m-%d").date() for d in dates]

    return subject_ids, dates, data


def get_embeddings_from_file(model_type, mask_pct, method="mean", encoder="cnn"):
    """
    Final embeddings obtained via mean pool over sequence length to get single embedding per day
    
    :param model_type: s4, mamba
    :param mask_pct: (int) 10, 30, 50, 70
    """
    extension = f"_{method}"
    if encoder is not None: extension += f"_{encoder}"

    embeddings_dir = f"/data1/emilyzho/tiles-2018-processed/tiles-test/embeddings/{model_type}/masking_{mask_pct}{extension}"
    embeddings_file = os.path.join(embeddings_dir, "embeddings.npy")
    index_file = os.path.join(embeddings_dir, "index.json")
    embeddings_list = np.load(embeddings_file)
    with open(index_file) as f:
        index = json.load(f)

    subject_ids = []
    dates = []
    embeddings = []

    for i in range(len(index)):
        entry = index[i]
        emb = embeddings_list[entry["row"]]  # shape (128,)
        subject_ids.append(entry["user"])
        date = entry["date"]
        date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
        dates.append(date)
        embeddings.append(emb)

    return subject_ids, dates, embeddings

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
        constants.Labels.SDNN,
        constants.Labels.RHR,
        constants.Labels.STEPS,
        constants.Labels.SLEEP_MINS
    ]
    debug = False


# Save test subject IDs to file -------------------------------------------------------------------
    # import json

    # subject_ids_open, dates_open, data_open = load_tiles_open(
    #     signal_columns=signal_columns,
    #     scale=scale, window_size=window_size, debug=debug
    # )
    # subject_ids_holdout, dates_holdout, data_holdout = load_tiles_holdout(
    #     signal_columns=signal_columns,
    #     scale=scale, window_size=window_size, debug=debug
    # )
    # df = pd.DataFrame({
    #     'ID': subject_ids_open+subject_ids_holdout,
    #     'Date': dates_open+dates_holdout
    # })

    # test_ratio = 1 / 11
    # gss = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=42)
    # train_indices, test_indices = next(gss.split(df, groups=df['ID']))
    # train_df = df.iloc[train_indices]
    # test_df = df.iloc[test_indices]

    # pretrain_subject_ids = train_df['ID'].tolist()
    # pretrain_dates = train_df['Date'].tolist()
    # test_subject_ids = test_df['ID'].tolist()
    # test_dates = test_df['Date'].tolist()

    # train_set = pd.DataFrame({'ID': pretrain_subject_ids, 'Date': pretrain_dates})
    # test_set = pd.DataFrame({'ID': test_subject_ids, 'Date': test_dates})
    # out = {
    #     "pretrain": {},
    #     "test": {}
    # }

    # unique_pretraining_subjects = train_set['ID'].unique().tolist()
    # for subject_id in unique_pretraining_subjects:
    #     dates = train_set[train_set['ID'] == subject_id]['Date'].tolist()
    #     dates = [d.strftime("%Y-%m-%d") for d in dates]
    #     out["pretrain"][subject_id] = dates

    # unique_test_subjects = test_set['ID'].unique().tolist()
    # for subject_id in unique_test_subjects:
    #     dates = test_set[test_set['ID'] == subject_id]['Date'].tolist()
    #     dates = [d.strftime("%Y-%m-%d") for d in dates]
    #     out["test"][subject_id] = dates

    # with open("/data1/emilyzho/tiles-2018-processed/subject_splits.json", "w") as f:
    #     json.dump(out, f, indent=4)

# Test updated label generation functions
    # debug = False
    # subject_ids, dates, data = get_data_from_splits()
    # label_types = ['NumberSteps', 'RestingHeartRate', 'SleepMinutesAsleep']
    
    # all_labels = generate_continuous_labels_day(subject_ids, dates, label_types=label_types)
    # for label_type in label_types:
    #     labels = all_labels[label_type]
    #     nan_indices = [i for i in range(len(labels)) if np.isnan(labels[i])]
    #     nan_indices.sort(reverse=True)
    #     for i in nan_indices:
    #         subject_ids.pop(i)
    #         data.pop(i)
    #         labels.pop(i)

    #     print(labels[0:100])

    # for label_type in ['shift']:
    #     labels = generate_
    # s(subject_ids, dates, label_type=label_type)
    #     print(labels[0:200]) 

# Test loading directly from embedding .npy files
    # subject_ids, dates, data = get_embeddings_from_file("s4", 30, method="raw", encoder="cnn")

    # binary_labels = ["age", "shift", "anxiety", "stress"]
    # continuous_labels = ["NumberSteps", "RestingHeartRate", "SleepMinutesAsleep"]

    # for label_type in binary_labels[0:1]:
    #     logging.info(f"Label: {label_type} " + "-"*60)
    #     labels = generate_binary_labels(subject_ids, dates, label_type=label_type)
    #     subject_ids = np.asarray(subject_ids)
    #     data = np.asarray(data)
    #     labels = np.asarray(labels)

    #     unique_values, counts = np.unique(labels, return_counts=True)
    #     print("Unique values:", unique_values)
    #     print("Counts:", counts)
    #     print(f"Embedding shape: {data[0].shape}")

    subject_ids, dates, data = get_data_from_splits("test")
    train_dataset = pd.DataFrame({
        'ID': subject_ids,
        'Date': dates,
        'Data': data
    })

    gss = GroupShuffleSplit(n_splits=1, test_size=1/2, random_state=42)
    train_indices, test_indices = next(gss.split(train_dataset, groups=train_dataset['ID']))
    train_df = train_dataset.iloc[train_indices]
    test_df = train_dataset.iloc[test_indices]

    pretrain_subject_ids = train_df['ID'].tolist()
    pretrain_dates = train_df['Date'].tolist()
    pretrain_data = train_df['Data'].tolist()
    print(type(pretrain_data[0]))
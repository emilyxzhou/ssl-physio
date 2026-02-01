import os
import pandas as pd
import sys
from pathlib import Path
USER_ROOT = str(Path(__file__).resolve().parents[3])
paths = [
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

import glob
import json
import numpy as np
import pprint

from pathlib import Path
from sklearn.preprocessing import StandardScaler

from tiles_dataloader import get_embeddings_from_file, get_data_from_splits, generate_binary_labels, generate_continuous_labels_day


def get_results_stats():
    results_dict = {
        "age": {
            "Model": [],
            "bACC": [],
            "ACC": [],
            "F1": [],
            "AUC": []
        },
        "shift": {
            "Model": [],
            "bACC": [],
            "ACC": [],
            "F1": [],
            "AUC": []
        },
        "anxiety": {
            "Model": [],
            "bACC": [],
            "ACC": [],
            "F1": [],
            "AUC": []
        },
        "stress": {
            "Model": [],
            "bACC": [],
            "ACC": [],
            "F1": [],
            "AUC": []
        }
    }

    results_files = glob.glob('/home/emilyzho/ssl-physio/results/*/classification/*.json')
    for file in results_files:
        file_metadata = file.split('/')[-1].split('.')[0].rsplit('_', 1)
        model_name = file_metadata[0]
        if "_probe" in model_name: model_name = model_name.replace("_probe", "probe")
        label = file_metadata[1]
        with open(file, 'r') as f:
            res = json.load(f)
            for col in results_dict[label].keys():
                if col == "Model": 
                    results_dict[label][col].append(model_name)
                else:
                    folds = res["test"][col]
                    avg_metric = round(np.mean(folds), 3)
                    std_metric = round(np.std(folds), 3)
                    res_string = f"{avg_metric} ({std_metric})"
                    results_dict[label][col].append(res_string)
    
    for label in results_dict.keys():
        print("="*90)
        print(label)
        print("="*90)

        results_df = pd.DataFrame(results_dict[label])
        results_df["Mask %"] = results_df["Model"].apply(lambda s: s.split("_")[1] if "_" in s else "None")
        results_df["Model type"] = results_df["Model"].apply(lambda s: s.split("_")[0])
        results_df["Method"] = results_df["Model"].apply(lambda s: s.rsplit("_", 1)[-1] if "_" in s else "None")
        results_df = results_df.drop(columns=["Model"])

        # results_df = results_df[results_df["Method"] != "finetune"]

        order = ["Model type", "Mask %", "Method", "bACC", "ACC", "F1", "AUC"]
        results_df = results_df[order]
        results_df = results_df.sort_values(by=["Model type", "Mask %", "Method"]).reset_index(drop=True)

        with pd.option_context('display.max_rows', None):
            print(results_df)



    results_dict = {
        "NumberSteps": {
            "Model": [],
            "MSE": [],
            "MAE": [],
            "R": []
        },
        "RestingHeartRate": {
            "Model": [],
            "MSE": [],
            "MAE": [],
            "R": []
        },
        "SleepMinutesAsleep": {
            "Model": [],
            "MSE": [],
            "MAE": [],
            "R": []
        }
    }

    results_files = glob.glob('/home/emilyzho/ssl-physio/results/*/regression/*.json')
    for file in results_files:
        file_metadata = file.split('/')[-1].split('.')[0].rsplit('_', 1)
        model_name = file_metadata[0]
        if "_probe" in model_name: model_name = model_name.replace("_probe", "probe")
        label = file_metadata[1]
        with open(file, 'r') as f:
            res = json.load(f)
            for col in results_dict[label].keys():
                if col == "Model": 
                    results_dict[label][col].append(model_name)
                else:
                    folds = res["test"][col]
                    avg_metric = round(np.mean(folds), 3)
                    std_metric = round(np.std(folds), 3)
                    res_string = f"{avg_metric} ({std_metric})"
                    results_dict[label][col].append(res_string)
    
    for label in results_dict.keys():
        print("="*90)
        print(label)
        print("="*90)

        results_df = pd.DataFrame(results_dict[label])
        results_df["Mask %"] = results_df["Model"].apply(lambda s: s.split("_")[1] if "_" in s else "None")
        results_df["Model type"] = results_df["Model"].apply(lambda s: s.split("_")[0])
        results_df["Method"] = results_df["Model"].apply(lambda s: s.rsplit("_", 1)[-1] if "_" in s else "None")
        results_df = results_df.drop(columns=["Model"])

        # results_df = results_df[results_df["Method"] != "finetune"]

        order = ["Model type", "Mask %", "Method", "MSE", "MAE", "R"]
        results_df = results_df[order]
        results_df = results_df.sort_values(by=["Model type", "Mask %", "Method"]).reset_index(drop=True)

        with pd.option_context('display.max_rows', None):
            print(results_df)


def get_label_stats():
    binary_labels = ["age", "shift", "anxiety", "stress"]
    df = {
        "ID": None,
        "shift": None,
        "age": None,
        "anxiety": None,
        "stress": None
    }

    subject_ids, dates, _ = get_data_from_splits("test")
    subject_ids = np.asarray(subject_ids)
    df["ID"] = subject_ids

    for label_type in binary_labels:
        labels = generate_binary_labels(subject_ids, dates, label_type=label_type, verbose=False)
        labels = np.asarray(labels)
        df[label_type] = labels

    df = pd.DataFrame(df)
    with pd.option_context('display.max_rows', None): print(df)


if __name__ == "__main__":
    get_results_stats()
    # get_label_stats()
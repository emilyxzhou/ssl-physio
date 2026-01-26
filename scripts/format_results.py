import glob
import json
import numpy as np
import os
import pandas as pd
import re

from pathlib import Path
USER_ROOT = str(Path(__file__).resolve().parents[2])


if __name__ == "__main__":
    SSL_ROOT = os.path.join(USER_ROOT, "ssl-physio")
    BASELINE_RESULTS_FOLDER = os.path.join(SSL_ROOT, "results", "baselines")
    SSL_RESULTS_FOLDER = os.path.join(SSL_ROOT, "results", "downstream")

    LABEL_MAPPINGS = {
        "age": "Age",
        "shift": "Shift",
        "anxiety": "Anxiety",
        "stress": "Stress",
        "bpm": "Average HR (next day)",
        "ms": "SDNN (next day)",
        "NumberSteps": "Total steps (next day)",
        "RestingHeartRate": "RHR (next day)",
        "SleepMinutesAsleep": "TST"
    }

    classification_results_files = glob.glob(
        os.path.join(SSL_ROOT, "results", "*", "classification", "*")
    )

    classification_results = {
        "model_type": [],
        "label_type": [],
        "ACC": [],
        "bACC": [],
        "F1": [],
        "AUC": []
    }

    for file in classification_results_files:
        metadata = file.split("/")[-1].split(".json")[0].rsplit("_", 1)

        model_type = metadata[0]
        label_type = metadata[1]
        label_type = LABEL_MAPPINGS[label_type]

        classification_results["model_type"].append(model_type)
        classification_results["label_type"].append(label_type)

        with open(file) as f:
            results = json.load(f)
            acc_mean = round(np.mean(results["test"]["ACC"]), 3)
            acc_std = round(np.std(results["test"]["ACC"]), 3)
            bacc_mean = round(np.mean(results["test"]["bACC"]), 3)
            bacc_std = round(np.std(results["test"]["bACC"]), 3)
            f1_mean = round(np.mean(results["test"]["F1"]), 3)
            f1_std = round(np.std(results["test"]["F1"]), 3)
            auc_mean = round(np.mean(results["test"]["AUC"]), 3)
            auc_std = round(np.std(results["test"]["AUC"]), 3)

            classification_results["ACC"].append(f"{acc_mean} ({acc_std})")
            classification_results["bACC"].append(f"{bacc_mean} ({bacc_std})")
            classification_results["F1"].append(f"{f1_mean} ({f1_std})")
            classification_results["AUC"].append(f"{auc_mean} ({auc_std})")

    classification_results = pd.DataFrame(classification_results).sort_values(by=["model_type", "label_type"]).reset_index(drop=True)

    with pd.option_context('display.max_rows', None):
        print(classification_results)

    # ----------------------------------------------------------------------------------------------------

    regression_results_files = glob.glob(
        os.path.join(SSL_ROOT, "results", "*", "regression", "*")
    )

    regression_results = {
        "model_type": [],
        "label_type": [],
        "MAE": [],
        "MSE": [],
        "R": []
    }

    for file in regression_results_files:
        metadata = file.split("/")[-1].split(".json")[0].rsplit("_", 1)

        model_type = metadata[0]
        if "SDNN" in model_type: 
            model_type = model_type.rsplit("_", 1)[0]
        label_type = metadata[1]
        label_type = LABEL_MAPPINGS[label_type]


        regression_results["model_type"].append(model_type)
        regression_results["label_type"].append(label_type)

        with open(file) as f:
            results = json.load(f)
            mae_mean = round(np.mean(results["test"]["MAE"]), 3)
            mae_std = round(np.std(results["test"]["MAE"]), 3)
            mse_mean = round(np.mean(results["test"]["MSE"]), 3)
            mse_std = round(np.std(results["test"]["MSE"]), 3)
            r_mean = round(np.mean(results["test"]["R"]), 3)
            r_std = round(np.std(results["test"]["R"]), 3)

            regression_results["MAE"].append(f"{mae_mean} ({mae_std})")
            regression_results["MSE"].append(f"{mse_mean} ({mse_std})")
            regression_results["R"].append(f"{r_mean} ({r_std})")

    regression_results = pd.DataFrame(regression_results).sort_values(by=["model_type", "label_type"]).reset_index(drop=True)

    with pd.option_context('display.max_rows', None):
        print(regression_results)
import json
import numpy as np
import torch
import random

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import DataLoader, Subset

from mamba.mamba_mae import MambaMAE
from s4_models.s4_mae import S4MAE


def split_k_fold(subject_ids, data, labels, num_folds=5, seed=37):
    random.seed(seed)
    unique_subjects = list(set(subject_ids))
    subjects_per_fold = (len(unique_subjects) // num_folds) + 1
    random.shuffle(unique_subjects)

    fold_ids = {subject_id: None for subject_id in unique_subjects}
    subject_id_folds = [[] for _ in range(num_folds)]
    data_folds = [[] for _ in range(num_folds)]
    labels_folds = [[] for _ in range(num_folds)]
    for i in range(num_folds):
        subject_sublist = unique_subjects[i*subjects_per_fold:i*subjects_per_fold + subjects_per_fold]
        for subject_id in subject_sublist:
            fold_ids[subject_id] = i

    for subject_id in fold_ids.keys():
        fold = fold_ids[subject_id]
        for i in range(len(subject_ids)):
            if subject_ids[i] == subject_id:
                subject_id_folds[fold].append(subject_ids[i])
                data_folds[fold].append(data[i])
                labels_folds[fold].append(labels[i])
    
    return subject_id_folds, data_folds, labels_folds


def stratified_group_split(subject_ids, input_data, labels, train_size=0.90, random_state=42):
    # Convert identifiers to numpy for masking logic
    subj_arr = np.array(subject_ids)
    label_arr = np.array(labels)

    # 1. Map each unique subject to its label for stratification
    unique_subjects, unique_indices = np.unique(subj_arr, return_index=True)
    unique_subject_labels = label_arr[unique_indices]

    # 2. Split the UNIQUE subjects (14% train, 86% test)
    train_subs, test_subs = train_test_split(
        unique_subjects,
        train_size=train_size,
        stratify=unique_subject_labels,
        random_state=random_state
    )

    # 3. Get the integer indices for the train and test sets
    train_indices = np.where(np.isin(subj_arr, train_subs))[0]
    test_indices = np.where(np.isin(subj_arr, test_subs))[0]

    # 4. Reconstruct lists using the original 'input_data' objects
    # This guarantees the internal elements remain np.ndarrays
    train_data = [input_data[i] for i in train_indices]
    test_data = [input_data[i] for i in test_indices]
    
    # Do the same for IDs and Labels for consistency
    train_subject_ids = [subject_ids[i] for i in train_indices]
    test_subject_ids = [subject_ids[i] for i in test_indices]
    
    train_labels = [labels[i] for i in train_indices]
    test_labels = [labels[i] for i in test_indices]

    return (train_subject_ids, test_subject_ids, 
            train_data, test_data, 
            train_labels, test_labels)


def get_kfold_loaders(dataloader, device='cuda', k=5):
    dataset = dataloader.dataset
    # Use the same parameters as your original loader for consistency
    loader_args = {
        "batch_size": dataloader.batch_size,
        "num_workers": dataloader.num_workers,
        "pin_memory": dataloader.pin_memory
    }

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    folds = []

    for train_idx, val_idx in kf.split(range(len(dataset))):
        # Create Subsets for this specific fold
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # Create new DataLoaders
        train_loader = DataLoader(train_subset, shuffle=True, generator=torch.Generator(device=device), **loader_args)
        val_loader = DataLoader(val_subset, shuffle=False, generator=torch.Generator(device=device), **loader_args)
        
        folds.append((train_loader, val_loader))
    
    return folds

# Usage:
# folds = get_kfold_loaders(your_existing_dataloader, k=5)
# for fold_idx, (train_loader, val_loader) in enumerate(folds):
#     train_eval_loop(fold_idx, train_loader, val_loader)


def normalize_list(x):
    min_x = min(x)
    max_x = max(x)
    return [(i - min_x) / (max_x - min_x) for i in x]


def load_model(checkpoint_path, config_path, model_type, mask_ratio=None, device="cuda:0"):
    # Read arguments -----------------------------------------------------------------------------------------------
    config = json.load(open(config_path, "r"))
    model_params = config["model_params"]
    if model_params["dec_hidden_dims"] is not None: model_params["d_model"] = model_params["dec_hidden_dims"][0]
    model_params["mask_ratio"] = 0.0

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if model_type == "s4":
        model = S4MAE(
            **model_params,
            classification="lin_probe"
        ).to(device)
    elif model_type == "mamba":
        model = MambaMAE(
            **model_params,
            classification="lin_probe"
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


if __name__ == "__main__":
    pass
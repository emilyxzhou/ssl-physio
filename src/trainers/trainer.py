import glob
import os
import pandas as pd
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt

from copy import deepcopy

os.environ["S4_FAST_CAUCHY"] = "0"
os.environ["S4_FAST_VAND"] = "0"
os.environ["S4_BACKEND"] = "keops"   # or "keops" if you installed pykeops

# Define logging console
import logging
logging.basicConfig(
    format="%(asctime)s %(levelname)-3s ==> %(message)s", 
    level=logging.INFO, 
    datefmt="%Y-%m-%d %H:%M:%S"
)


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


class Trainer:
    def __init__(
            self, n_epochs, checkpoint_dir, checkpoint_prefix, model_save_folder,
            mode="full", reconstruction="full",
            use_wandb=False,
            n_iters=None,
        ):
        self.n_epochs = n_epochs
        self.n_iters = n_iters if n_iters is not None else 10_000
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = checkpoint_prefix
        self.model_save_folder = model_save_folder

        self.start_epoch, self.start_iter = 0, 0

        self.train_losses = []
        self.val_losses = []
        
        self.mode = mode
        self.reconstruction = reconstruction

        self.use_wandb = use_wandb


    def train_recon(
        self, model, train_dataloader, val_dataloader,
        optimizer, criterion=nn.MSELoss(),
        scheduler=None, 
        mask_ratio=0.0,
        resume_checkpoint: str=None,
        device: str="cuda", debug=False
    ):
        print("Starting training...")
        model = model.to(device)
        if resume_checkpoint:
            model = self.resume_checkpoint(resume_checkpoint, model)

        for epoch in range(self.start_epoch, self.n_epochs):
            logging.info(f"Epoch {epoch+1}/{self.n_epochs}")

            model.train()
            total_loss = 0.0
            total_samples = 0

            for step, batch_data in enumerate(train_dataloader):
                optimizer.zero_grad()

                # Transfer to GPU
                batch, _, _ = batch_data
                batch = batch.to(device)                  # original shape (batch_size, 1440, num_features)
                batch = torch.transpose(batch, 1, 2)      # new shape (batch_size, num_features, 1440)
                if self.mode == "heart rate": 
                    batch = batch[:, 0, :]    # Heart rate only
                    batch = torch.unsqueeze(batch, 1)
                elif self.mode == "step count": 
                    batch = batch[:, 1, :]    # Step count only
                    batch = torch.unsqueeze(batch, 1)

                out, target, mask = model(batch, mask_ratio=mask_ratio)
                mask = mask.to(int)
                # Get masked positions
                x_recon_masked = out * (1-mask)
                x_masked = target * (1-mask)
                if mask_ratio == 0: 
                    if self.reconstruction == "full": loss = criterion(out, target)
                    else: loss = criterion(out[:, 0, :], target[:, 0, :])    # Only learn heart rate reconstruction
                else:
                    if self.reconstruction == "full": loss = criterion(x_recon_masked, x_masked)
                    else: loss = criterion(x_recon_masked[:, 0, :], x_masked[:, 0, :])    # Only learn heart rate reconstruction

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item() * batch.size(0)
                total_samples += batch.size(0)

                if ((step+1)%100 == 0 or step == len(train_dataloader)-1):
                    logging.info(f"Epoch {epoch+1}, step {step+1} | loss {loss.item()}")
                    if self.use_wandb:
                        wandb.log({
                            "train_loss": loss.item()
                        })

            if scheduler is not None: scheduler.step()

            avg_loss = total_loss / total_samples
            logging.info(f"Train loss: {avg_loss:.4f}")
            self.train_losses.append(avg_loss)

            avg_val_loss = self.validate_recon(model, val_dataloader, criterion, device=device)
            self.val_losses.append(avg_val_loss)

            if (epoch+1) % 10 == 0 and not debug:
                checkpoint_filename = f"{self.checkpoint_prefix}_epoch_{epoch+1}.pt"
                checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_filename)
                torch.save(model.state_dict(), checkpoint_path)

        return model
    

    def validate_recon(
        self, model, val_dataloader,
        criterion=nn.MSELoss(),
        mask_ratio=0.0,
        split="val",
        device: str="cuda",
    ):
        model.eval()
        total_val_loss = 0.0
        total_val_samples = 0

        with torch.no_grad():
            for step, batch_data in enumerate(val_dataloader):
                # Transfer to GPU
                batch, _, _ = batch_data
                batch = batch.to(device)                  # original shape (batch_size, 1440, num_features)
                batch = torch.transpose(batch, 1, 2)      # new shape (batch_size, num_features, 1440)
                if self.mode == "heart rate": 
                    batch = batch[:, 0, :]    # Heart rate only
                    batch = torch.unsqueeze(batch, 1)
                elif self.mode == "step count": 
                    batch = batch[:, 1, :]    # Step count only
                    batch = torch.unsqueeze(batch, 1)

                out, target, mask = model(batch, mask_ratio=mask_ratio)
                mask = mask.to(int)
                # Get masked positions
                x_recon_masked = out * (1-mask)
                x_masked = target * (1-mask)
                if mask_ratio == 0: 
                    if self.reconstruction == "full": loss = criterion(out, target)
                    else: loss = criterion(out[:, 0, :], target[:, 0, :])    # Only learn heart rate reconstruction
                else:
                    if self.reconstruction == "full": loss = criterion(x_recon_masked, x_masked)
                    else: loss = criterion(x_recon_masked[:, 0, :], x_masked[:, 0, :])    # Only learn heart rate reconstruction

                total_val_loss += loss.item() * batch.size(0)
                total_val_samples += batch.size(0)

                if step == len(val_dataloader)-1:
                    logging.info(f"[{split}] Step {step+1} | Loss {loss.item():.4f}")
                    if self.use_wandb:
                        wandb.log({
                            f"{split}_loss": loss.item()
                        })

        avg_val_loss = total_val_loss / total_val_samples
        logging.info(f"Validation loss: {avg_val_loss:.4f}")
        return avg_val_loss
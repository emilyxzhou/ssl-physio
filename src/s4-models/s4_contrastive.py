# Contrastive Learning with S4 models pipeline

import torch
import torch.nn as nn
import torch.nn.functional as F 

import numpy as np

from s4model import S4Model
from s4_mae import ConvEncoder
import random

class ContrastiveAugmenter:
    def __init__(self, probs: dict, max_shift: int = 120, patch_size: int = 30, cutout_size: int = 60):
        """
        probs: dict of probabilities for each augmentation. Example:
            probs = {'patch_mask': 0.5, 'time_shift': 0.5, 'cutout': 0.5}
        max_shift: max absolute value for time shift
        patch_size: size of each patch for patch masking
        cutout_size: size of cutout region
        """
        self.probs = probs
        self.max_shift = max_shift
        self.patch_size = patch_size
        self.cutout_size = cutout_size

        # Here, dim 0 is heart rate and hence is set to the lower end of z-score
        # for step count it was set to a negative value to indicate -1 is masking
        self.mask_dict = {0: -3.0, 1: -1.0}

    def patch_mask(self, x):
        B, T, C = x.shape
        x_aug = x.clone()
        for b in range(B):
            for c in range(C):
                if random.random() < self.probs.get('patch_mask', 0):
                    start = random.randint(0, T - self.patch_size)
                    x_aug[b, start:start+self.patch_size, c] = self.mask_dict[c]
        return x_aug

    def time_shift(self, x):
        B, T, C = x.shape
        x_aug = x.clone()
        for b in range(B):
            if random.random() < self.probs.get('time_shift', 0):
                shift = random.randint(-self.max_shift, self.max_shift)
                if shift > 0:
                    x_aug[b, shift:, :] = x[b, :-shift, :]
                    x_aug[b, :shift, :] = 0
                elif shift < 0:
                    shift = -shift
                    x_aug[b, :-shift, :] = x[b, shift:, :]
                    x_aug[b, -shift:, :] = 0
        return x_aug

    def cutout(self, x):
        B, T, C = x.shape
        x_aug = x.clone()
        for b in range(B):
            if random.random() < self.probs.get('cutout', 0):
                start = random.randint(0, T - self.cutout_size)
                x_aug[b, start:start+self.cutout_size, :] = 0
        return x_aug

    def __call__(self, x):
        """Randomize augmentation order and apply sequentially"""
        x_aug = x.clone()
        # list of augmentation functions
        aug_functions = [self.patch_mask, self.time_shift]
        random.shuffle(aug_functions)  # random order each call
        for aug in aug_functions:
            x_aug = aug(x_aug)
        return x_aug

class S4Contrastive(nn.Module):

    def __init__(self,
                 d_model: int=None,
                 d_input: int=2,
                 d_output: int=256,
                 enc_hidden_dims: list=[32, 64, 128],
                 n_layers_s4: int=4,
                 classification: bool=False,
                 verbose: bool=False):

        super().__init__()

        # Convolution encoder
        if enc_hidden_dims is not None:
            self.encoder = ConvEncoder(
                input_channels=d_input,
                hidden_dims=enc_hidden_dims,
                verbose=verbose
            )
            s4_dim = enc_hidden_dims[-1]
        else:
            self.encoder = nn.Identity()
            s4_dim = d_input
        
        # S4 sequence to sequence model
        self.s4_model = S4Model(
            d_input=s4_dim,
            d_output=d_output,
            d_model=d_model,
            dropout=0.3,
            prenorm=False,
            n_layers=n_layers_s4
        )

        # TODO MLP projection head for contrastive learning
        self.mlp_projection_head = nn.Sequential(nn.Linear(d_output, 4*d_output),
                                                 nn.GELU(),
                                                 nn.Linear(4*d_output, d_output))

        # TODO implementation of classification head
        if classification:
            raise NotImplementedError
        
        self.verbose = verbose
     
    def forward(self, x, training: bool=True):
        """
            x: (B, 1440, n_signals)
                Here, B is batch size
                    1440 refers to minute level data per day
                    n_signals is the number of signals like Heart Rate and Step Count
            Return:
                MLP projection output
        """
        
        # conv_output -> (B, enc_dim[-1], L)
        conv_output = self.encoder(x)
        if self.verbose: print(f"Conv output shape: {conv_output.shape}")

        # s4_output -> (B, L, d_output)
        s4_output = self.s4_model(conv_output.transpose(-1, -2))
        if self.verbose: print(f"S4 output shape: {s4_output.shape}")

        # mean pool along the length dimension
        # s4_embedding -> (B, d_output)
        s4_embedding = s4_output.mean(dim=1)
        if self.verbose: print(f"S4 embedding output shape: {s4_embedding.shape}")

        # pass through the MLP projection head for contrastive loss computation
        # z -> (B, d_output)
        z = self.mlp_projection_head(s4_embedding)
        if self.verbose: print(f"MLP Projection head output shape: {z.shape}")

        if training:
            return z
        

if __name__ == "__main__":
    model = S4Contrastive(d_model=256, d_input=2, d_output=512, verbose=True)
    dummy_input = torch.randn(32, 2, 1440)

    model(dummy_input)
    print(f"Number of training parameters: {sum([p.numel() for p in model.parameters()])}")
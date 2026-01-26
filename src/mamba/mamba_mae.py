# MAMBA MAE implementation
import os, sys 
from pathlib import Path
USER_ROOT = str(Path(__file__).resolve().parents[3])

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np 

from torchinfo import summary

sys.path.append(os.path.join(USER_ROOT, "ssl-physio", "src", "s4-models"))
from linear_classifier import CNN, LogisticRegressionHead
from regressor import Regressor

from mamba_ssm import Mamba

class PatchMasking(nn.Module):
    def __init__(self, ratio: float=0.3, patch_size: float=0.02, device: str="cuda"):
        super().__init__()

        self.ratio = ratio
        self.patch_size = patch_size
        self.device = device

        self.num_patches_masked = int(round(self.ratio/self.patch_size))

        self.patch_indices = np.linspace(0, 1, int(round(1/self.patch_size)), endpoint=False)
    
    def __generate_mask__(self, shape_mask):
        timestamps = shape_mask[-1]
        mask = torch.ones(shape_mask)

        if self.ratio == 0.0:
            return mask.to(self.device)
        
        # Generate the patches to mask
        indices_mask = np.random.choice(self.patch_indices, self.num_patches_masked, replace=False)

        for index_mask in indices_mask:
            start_mask, end_mask = int(timestamps*index_mask), int(timestamps*(index_mask + self.patch_size))
            mask[:, :, start_mask:end_mask] = 0.0 
        return mask.to(self.device)

    def forward(self, x):
        mask = self.__generate_mask__(x.shape)
        masked_x = mask * x
        return mask, masked_x


class BlockConv(nn.Module):
    def __init__(
        self, in_channel: int=512, out_channel: int=512, 
        kernel_size: int=3, stride: int=2, padding: int=1, output_padding: int=1,
        norm=nn.GroupNorm, dropout: float=0.3, residual: bool=False,
        act_layer = nn.GELU, conv: str="conv"
    ):
        """A single convolution block with normalization and dropout"""
        super().__init__()

        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.padding = padding
        self.output_padding = output_padding
        self.stride = stride
        self.dropout = dropout

        if residual is not False:
            raise NotImplementedError

        if conv == "conv":
            self.conv_layer = nn.Conv1d(
                in_channels=self.in_channel,
                out_channels=self.out_channel,
                kernel_size=self.kernel_size,
                stride=self.stride, padding=self.padding)
        else:
            self.conv_layer = nn.ConvTranspose1d(
                in_channels=self.in_channel,
                out_channels=self.out_channel,
                kernel_size=self.kernel_size,
                stride=self.stride, padding=self.padding,
                output_padding=self.output_padding
            )
        
        self.norm_layer = norm(1, out_channel)

        self.dropout_layer = nn.Dropout1d(self.dropout)

        if act_layer is not None:
            self.act_layer = act_layer()
        else:
            self.act_layer = None

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.dropout_layer(x)
        x = self.norm_layer(x)

        if self.act_layer is not None:
            x = self.act_layer(x)
        return x


class BlockDecoderLast(BlockConv):
    def __init__(
            self, 
            in_channel: int=512, out_channel: int=512, 
            kernel_size: int=3, stride: int=2, padding: int=1, output_padding: int=1, 
            norm=nn.GroupNorm, dropout: float=0.3, residual: bool=False, 
            act_layer=nn.GELU, conv: str="deconv"
    ):
        super().__init__(
            in_channel, out_channel, 
            kernel_size, stride, padding, output_padding, 
            norm, dropout, residual, act_layer, conv
        )
    
    def forward(self, x):
        x = self.conv_layer(x)
        return x
    

class ConvEncoder(nn.Module):
    def __init__(
        self, input_channels: int=2,
        hidden_dims=[32, 64, 128],
        kernel_size: int=3, stride: int=2, padding: int=1, output_padding: int=1,
        norm=nn.GroupNorm, dropout: float=0.3, residual: bool=False,
        act_layer = nn.GELU,
        verbose=False
    ):

        super().__init__()

        self.block_layers = nn.ModuleList()

        self.block_layers.append(BlockConv(
            input_channels, hidden_dims[0],
            kernel_size, stride, padding, output_padding,
            norm, dropout, residual, act_layer
        ))

        for i in range(1, len(hidden_dims)):
            self.block_layers.append(BlockConv(
            hidden_dims[i-1], hidden_dims[i],
            kernel_size, stride, padding, output_padding,
            norm, dropout, residual, act_layer
        ))

        self.verbose = verbose
        
    def forward(self, x):
        for i, block_layer in enumerate(self.block_layers):
            x = block_layer(x)
            if self.verbose: print(f"ConvEncoder layer {i+1} output shape: {x.shape}")
        return x
    

class ConvDecoder(nn.Module):

    def __init__(
        self, input_channels: int=2,
        hidden_dims=[128, 64, 32],
        kernel_size: int=3, stride: int=2, padding: int=1, output_padding: int=1,
        norm=nn.GroupNorm, dropout: float=0.3, residual: bool=False,
        act_layer = nn.GELU,
        verbose=False
    ):
        super().__init__()

        self.block_layers = nn.ModuleList()
        
        for i in range(len(hidden_dims)-1):
            self.block_layers.append(BlockConv(
                hidden_dims[i], hidden_dims[i+1],
                kernel_size, stride, padding, output_padding,
                norm, dropout, residual, act_layer, "deconv"
            ))
        
        self.block_layers.append(BlockDecoderLast(
            hidden_dims[-1], input_channels,
            kernel_size, stride, padding, output_padding,
            norm, dropout, residual, None, "deconv"
        ))

        self.verbose = verbose
        
    def forward(self, x):
        for i, block_layer in enumerate(self.block_layers):
            x = block_layer(x)
            if self.verbose: print(f"ConvDecoder layer {i+1} output shape: {x.shape}")
        return x

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        # RMSNorm is often preferred for Mamba, but LayerNorm works too
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # Pre-Norm Residual connection pattern: x + Mamba(Norm(x))
        return x + self.mamba(self.norm(x))
    

class MambaModel(nn.Module):

    def __init__(
            self,
            d_input: int=None,
            d_model: int=None,
            d_output=None,
            d_state=16,
            expand=2,
            n_layers=6,
            pooling=False
    ):
        super().__init__()

        self.encoder = nn.Linear(d_input, d_model, bias=False)
        
        self.mamba_layers = nn.ModuleList()

        for _ in range(n_layers):
            self.mamba_layers.append(
                MambaBlock(d_model=d_model, d_state=d_state, expand=expand) 
            )
        
        if d_output is not None:
            # Linear decoder
            self.decoder = nn.Linear(d_model, d_output)
        else:
            self.decoder = nn.Identity()

        self.pooling = pooling

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        x = self.encoder(x)
        for layer in self.mamba_layers:
            x = layer(x)

        # Pooling: average pooling over the sequence length
        if self.pooling: x = x.mean(dim=1)

        # Decode the outputs
        x = self.decoder(x)  # (B, L, d_model) -> (B, L, d_output)
        
        return x

    
class MambaMAE(nn.Module):

    def __init__(
            self,
            d_model: int=None,
            d_input: int=2,
            d_output: int=256,
            enc_hidden_dims=[32, 64, 128],
            dec_hidden_dims=[128, 64, 32],
            n_layers_seq: int=6,
            mask_ratio=0.25,
            classification=False,
            verbose=False
    ):
        
        super(MambaMAE, self).__init__()
        self.mask_ratio = mask_ratio

        if d_model is None: d_model = dec_hidden_dims[0]

        if enc_hidden_dims is not None:
            self.encoder = ConvEncoder(
                input_channels=d_input,
                hidden_dims=enc_hidden_dims,
                verbose=verbose
            )
            enc_out_dim = enc_hidden_dims[-1]
            seq_dim = enc_hidden_dims[-1]
        else: 
            self.encoder = nn.Identity()
            enc_out_dim = d_input
            seq_dim = d_model

        self.mask = PatchMasking(mask_ratio)

        self.seq_model = MambaModel(
            d_input=enc_out_dim, d_model=d_model, d_state=16, expand=2, n_layers=n_layers_seq
        )

        self.norm_f = nn.LayerNorm(seq_dim)

        if dec_hidden_dims is not None:
            if enc_hidden_dims is None: 
                stride = 1
                output_padding = 0
            else: 
                stride = 2
                output_padding = 1
            self.decoder = ConvDecoder(
                input_channels=d_input,
                hidden_dims=dec_hidden_dims,
                stride=stride, output_padding=output_padding,
                verbose=verbose
            )
        else: 
            print("Setting decoder to identity mapping.")
            self.decoder = nn.Identity()
        
        if classification in ["finetune"]:
            dummy_input = torch.randn(1, d_input, 1440)
            conv_output = self.encoder(dummy_input.clone())
            seq_output = conv_output.transpose(1, 2)
            seq_output = self.seq_model(seq_output)
            seq_output = self.norm_f(seq_output)
            self.cls_head = CNN(
                d_input=seq_output.shape[1],
                sequence_len=seq_output.shape[2]
            )
        elif classification == "lin_probe":
            dummy_input = torch.randn(1, d_input, 1440)
            conv_output = self.encoder(dummy_input.clone())
            seq_output = conv_output.transpose(1, 2)
            seq_output = self.seq_model(seq_output)
            seq_output = self.norm_f(seq_output)
            self.cls_head = LogisticRegressionHead(
                d_input=seq_output.shape[1],
                sequence_len=seq_output.shape[2]
            )
        else:
            self.cls_head = None

        self.verbose = verbose
    
    def forward(self, x, mask_ratio=None):

        if mask_ratio is not None:
            self.mask = PatchMasking(mask_ratio)
        mask, masked_input = self.mask(x)

        if self.verbose: print(f"Masked input shape: {masked_input.shape}")

        conv_output = self.encoder(masked_input.clone())
        if self.verbose: print(f"Conv output shape: {conv_output.shape}")
        
        seq_output = conv_output.transpose(1, 2)
        seq_output = self.seq_model(seq_output)
        seq_output = self.norm_f(seq_output)

        if self.verbose: print(f"Mamba output shape: {seq_output.shape}")

        if self.cls_head is not None:
            logits = self.cls_head(seq_output)
            return logits
        else:
            decoder_out = self.decoder(seq_output.transpose(-1, -2).clone())
            return decoder_out, x, mask
        

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    debug = True 

    use_wandb = not debug
    verbose = True 

    mode = "full"
    reconstruction = "full"

    d_model = 128 
    if mode != "full": d_input = 1
    else: d_input = 2    

    d_output = 128

    # enc_hidden_dims = [32, 64, 128]
    enc_hidden_dims = None
    dec_hidden_dims = [128, 64, 32]

    n_layers_seq = 6

    mask_ratio = 0.25

    model = MambaMAE(
        d_model=dec_hidden_dims[0],
        d_input=d_input,
        d_output=d_output,
        enc_hidden_dims=enc_hidden_dims,
        dec_hidden_dims=dec_hidden_dims,
        n_layers_seq=n_layers_seq,
        mask_ratio=mask_ratio,
        classification=False,
        verbose=verbose
    ).to(device)

    summary(model, input_size=(1, d_input, 1440))







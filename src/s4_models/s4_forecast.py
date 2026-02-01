import numpy as np
import torch

from torch import nn
from torchinfo import summary

from binary_classifier import BinaryClassifier
from s4model import S4Model


class S4Forecast(nn.Module):

    def __init__(
            self,
            d_input: int=2, 
            d_model: int=128, 
            d_output: int=2,
            n_layers_s4: int=6,
            sequence_len=360,
            classification=False,
            device="cuda",
            verbose=False
        ):
        
        super().__init__()
        self.sequence_len = sequence_len

        self.s4_model = S4Model(
            d_input=d_input,
            d_output=d_output,
            d_model=d_model,
            n_layers=n_layers_s4,
            dropout=0.3,
            prenorm=False
        ).to(device)
        self.linear = nn.Linear(d_model, d_output)

        if classification:
            dummy_input = torch.randn(1, d_input, sequence_len).to(device)
            x = self.s4_model(dummy_input.transpose(-1, -2).clone())
            self.cls_head = BinaryClassifier(
                d_input=x.shape[1],
                sequence_len=x.shape[2]
            ).to(device)
        else:
            self.cls_head = None

        self.verbose = verbose

    def forward(self, x):
        out = self.s4_model(x.transpose(-1, -2).clone())
        if self.verbose: print(f"S4 output shape: {out.shape}")

        if self.cls_head is not None:
            out = self.cls_head(out)
        else:
            out = out.mean(dim=1)
            out = self.linear(out)
        
        return out
        

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    debug = True
    verbose = True

    lr = 5e-4

    mode = "full"

    # scale = "mean"
    # scale = "median"
    scale = "scale"

    d_model = 64
    d_input = 2
    d_output = 2

    n_layers_s4 = 4

    sequence_len = 360

    model = S4Forecast(
        d_input=d_input,
        d_model=d_model,
        d_output=d_output,
        n_layers_s4=n_layers_s4,
        sequence_len=sequence_len,
        classification=True,
        verbose=verbose
    ).to(device)
    summary(model, input_size=(1, d_input, 360))
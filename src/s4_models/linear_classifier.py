import numpy as np
import torch
import torch.nn.functional as F

from torch import nn
from torchinfo import summary


class CNN(nn.Module):

    def __init__(
        self,
        d_input: int, 
        sequence_len=1440,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=d_input, out_channels=128, kernel_size=3, padding="same")
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding="same")
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding="same")
        self.relu3 = nn.ReLU()
        self.pool3 = nn.AvgPool1d(kernel_size=2)

        self.conv4 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding="same")
        self.relu4 = nn.ReLU()
        self.pool4 = nn.AvgPool1d(kernel_size=2)

        dummy_input = torch.randn(1, d_input, sequence_len)
        x = self.pool1(self.relu1(self.conv1(dummy_input)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.pool4(self.relu4(self.conv4(x)))
        self._to_linear = x.shape[1] * x.shape[2]    # channels * final_sequence_length
        self.fc = nn.Linear(self._to_linear, 1)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.pool4(self.relu4(self.conv4(x)))
        x = x.view(-1, self._to_linear)
        x = self.fc(x)
        return x
    

class LogisticRegressionHead(nn.Module):

    def __init__(
        self,
        d_input: int, 
        sequence_len=1440,
    ):
        super().__init__()

        # Flatten
        dummy_input = torch.randn(1, d_input, sequence_len)
        self._to_linear = dummy_input.shape[1] * dummy_input.shape[2]    # channels * final_sequence_length

        self.fc = nn.Linear(self._to_linear, 1)

    def forward(self, x):
        x = x.view(-1, self._to_linear)    # Flatten tensor
        x = self.fc(x)
        return x
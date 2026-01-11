"""
Sequence prediction models for embedding sequencers.

Models take sequences of day embeddings and predict future biosignal values.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod


class EmbeddingSequenceModel(ABC, nn.Module):
    """
    Abstract base class for embedding sequence models.
    
    All models:
    - Input: (batch, days_given, embedding_dim=128)
    - Output: (batch, days_predicted, output_dim)
    
    output_dim depends on output_type:
    - 'bpm': 1 (avg heart rate)
    - 'steps': 1 (total step count)
    - 'both': 2 (avg_bpm, total_steps)
    """
    
    def __init__(self, days_given: int, days_predicted: int, output_type: str, embedding_dim: int = 128):
        super().__init__()
        self.days_given = days_given
        self.days_predicted = days_predicted
        self.output_type = output_type
        self.embedding_dim = embedding_dim
        
        # Determine output dimension
        if output_type == 'bpm':
            self.output_dim = 1
        elif output_type == 'steps':
            self.output_dim = 1
        elif output_type == 'both':
            self.output_dim = 2
        else:
            raise ValueError(f"Unknown output_type: {output_type}")
    
    @abstractmethod
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (batch, days_given, embedding_dim)
        
        Returns:
            (batch, days_predicted, output_dim)
        """
        pass
    
    def get_hyperparameters(self) -> dict:
        """Return model hyperparameters for logging."""
        return {
            'days_given': self.days_given,
            'days_predicted': self.days_predicted,
            'output_type': self.output_type,
            'embedding_dim': self.embedding_dim,
            'output_dim': self.output_dim
        }


class CNNModel(EmbeddingSequenceModel):
    """
    1D CNN for sequence prediction.
    
    Architecture:
    - Transpose to (batch, embedding_dim, days_given)
    - Conv1D layers with ReLU
    - Flatten and FC to output
    """
    
    def __init__(self, days_given: int, days_predicted: int, output_type: str, 
                 embedding_dim: int = 128, hidden_channels: int = 64):
        super().__init__(days_given, days_predicted, output_type, embedding_dim)
        
        self.hidden_channels = hidden_channels
        
        # Conv layers - kernel size adapts to sequence length
        kernel_size = min(3, days_given)
        
        self.conv1 = nn.Conv1d(embedding_dim, hidden_channels, kernel_size=kernel_size, padding='same')
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels // 2, kernel_size=kernel_size, padding='same')
        
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels // 2)
        
        # FC layers
        fc_input_dim = (hidden_channels // 2) * days_given
        self.fc1 = nn.Linear(fc_input_dim, 64)
        self.fc2 = nn.Linear(64, days_predicted * self.output_dim)
        
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # x: (batch, days_given, embedding_dim)
        # Transpose to (batch, embedding_dim, days_given) for Conv1d
        x = x.transpose(1, 2)
        
        # Conv layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        # Reshape to (batch, days_predicted, output_dim)
        x = x.view(-1, self.days_predicted, self.output_dim)
        
        return x
    
    def get_hyperparameters(self) -> dict:
        params = super().get_hyperparameters()
        params.update({
            'model_type': 'cnn',
            'hidden_channels': self.hidden_channels,
            'dropout': 0.2
        })
        return params


class NNModel(EmbeddingSequenceModel):
    """
    Multi-head feedforward neural network for sequence prediction.
    
    Architecture:
    - Flatten input to (batch, days_given * embedding_dim)
    - Shared hidden layers
    - Output layer with days_predicted * output_dim outputs
    """
    
    def __init__(self, days_given: int, days_predicted: int, output_type: str,
                 embedding_dim: int = 128, hidden_dim: int = 128):
        super().__init__(days_given, days_predicted, output_type, embedding_dim)
        
        self.hidden_dim = hidden_dim
        
        input_dim = days_given * embedding_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, days_predicted * self.output_dim)
        
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # x: (batch, days_given, embedding_dim)
        # Flatten to (batch, days_given * embedding_dim)
        x = x.view(x.size(0), -1)
        
        # Hidden layers (no BatchNorm - works with any batch size including 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        # Reshape to (batch, days_predicted, output_dim)
        x = x.view(-1, self.days_predicted, self.output_dim)
        
        return x
    
    def get_hyperparameters(self) -> dict:
        params = super().get_hyperparameters()
        params.update({
            'model_type': 'nn',
            'hidden_dim': self.hidden_dim,
            'dropout': 0.2,
            'batch_norm': False
        })
        return params


def create_model(model_type: str, days_given: int, days_predicted: int, output_type: str, 
                 embedding_dim: int = 128) -> EmbeddingSequenceModel:
    """
    Factory function to create models.
    
    Args:
        model_type: 'cnn' or 'nn'
        days_given: number of input days
        days_predicted: number of output days
        output_type: 'bpm', 'steps', or 'both'
        embedding_dim: dimension of embeddings (default 128)
    
    Returns:
        Model instance
    """
    if model_type == 'cnn':
        return CNNModel(days_given, days_predicted, output_type, embedding_dim)
    elif model_type == 'nn':
        return NNModel(days_given, days_predicted, output_type, embedding_dim)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


class CombinedLoss(nn.Module):
    """
    Combined loss for predicting both bpm and steps.
    
    Loss = bpm_weight * MSE(bpm) + steps_weight * MAE(steps)
    
    Weights are computed to equalize contribution based on variance.
    """
    
    def __init__(self, bpm_weight: float, steps_weight: float):
        super().__init__()
        self.bpm_weight = bpm_weight
        self.steps_weight = steps_weight
        # self.mse = nn.MSELoss()
        self.mse = nn.L1Loss()
    
    def forward(self, pred, target):
        """
        Args:
            pred: (batch, days_predicted, 2) - [bpm, steps]
            target: (batch, days_predicted, 2) - [bpm, steps]
        """
        bpm_loss = self.mse(pred[:, :, 0], target[:, :, 0])
        steps_loss = F.l1_loss(pred[:, :, 1], target[:, :, 1])
        
        return self.bpm_weight * bpm_loss + self.steps_weight * steps_loss


def get_loss_function(output_type: str, bpm_weight: float = 0.5, steps_weight: float = 0.5):
    """
    Get appropriate loss function for output type.
    
    Args:
        output_type: 'bpm', 'steps', or 'both'
        bpm_weight: weight for bpm loss (only used for 'both')
        steps_weight: weight for steps loss (only used for 'both')
    
    Returns:
        Loss function
    """
    if output_type == 'bpm':
        # return nn.MSELoss()
        return nn.L1Loss()
    elif output_type == 'steps':
        return nn.L1Loss()  # MAE
    elif output_type == 'both':
        return CombinedLoss(bpm_weight, steps_weight)
    else:
        raise ValueError(f"Unknown output_type: {output_type}")


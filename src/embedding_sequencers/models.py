"""
Sequence prediction models for embedding sequencers.

Models take sequences of day embeddings and predict future biosignal values.
Supports both regression (bpm, steps) and binary classification (anxiety, stress).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

# Output types and their properties
OUTPUT_TYPE_CONFIG = {
    'anxiety': {'output_dim': 1, 'task': 'binary', 'target_idx': 4},
    'stress': {'output_dim': 1, 'task': 'binary', 'target_idx': 5},
    'NumberSteps': {'output_dim': 1, 'task': 'regression', 'target_idx': 6},
    'RestingHeartRate': {'output_dim': 1, 'task': 'regression', 'target_idx': 7},
    'SleepMinutesAsleep': {'output_dim': 1, 'task': 'regression', 'target_idx': 8},
}

class EmbeddingSequenceModel(ABC, nn.Module):
    """
    Abstract base class for embedding sequence models.
    
    All models:
    - Input: (batch, days_given, embedding_dim=128)
    - Output: (batch, days_predicted, output_dim)
    
    output_type:
    - 'anxiety': 1 
    - 'stress': 1 
    - 'NumberSteps': 1 
    - 'RestingHeartRate': 1 
    - 'SleepMinutesAsleep': 1
    """
    
    def __init__(self, days_given: int, days_predicted: int, output_type: str,
                 embedding_dim: int, sequence_len: int):
        super().__init__()
        self.days_given = days_given
        self.days_predicted = days_predicted
        self.output_type = output_type
        self.embedding_dim = embedding_dim
        self.sequence_len = sequence_len
        
        # Determine output dimension and task type
        if output_type not in OUTPUT_TYPE_CONFIG:
            raise ValueError(f"Unknown output_type: {output_type}. Valid types: {list(OUTPUT_TYPE_CONFIG.keys())}")
        
        config = OUTPUT_TYPE_CONFIG[output_type]
        self.output_dim = config['output_dim']
        self.task_type = config['task']
        self.target_idx = config['target_idx']
    
    @abstractmethod
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (batch, days_given, sequence_len, embedding_dim)
        
        Returns:
            (batch, days_predicted, output_dim)
            For binary classification, these are logits (not probabilities).
        """
        pass
    
    def get_hyperparameters(self) -> dict:
        """Return model hyperparameters for logging."""
        return {
            'days_given': self.days_given,
            'days_predicted': self.days_predicted,
            'output_type': self.output_type,
            'task_type': self.task_type,
            'embedding_dim': self.embedding_dim,
            'sequence_len': self.sequence_len,
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
    

class CNN(EmbeddingSequenceModel):

    def __init__(
        self, days_given: int, days_predicted: int, output_type: str,
        embedding_dim: 128, sequence_len=180
    ):
        super().__init__(days_given, days_predicted, output_type, embedding_dim, sequence_len)

        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=3, padding="same")
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

        # FC layers
        fc_input_dim = (32 // 2) * days_given
        self.fc1 = nn.Linear(fc_input_dim, days_predicted * self.output_dim)
        
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.pool4(self.relu4(self.conv4(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = self.fc1(x)
        
        # Reshape to (batch, days_predicted, output_dim)
        x = x.view(-1, self.days_predicted, self.output_dim)
        
        return x
    
    def get_hyperparameters(self) -> dict:
        params = super().get_hyperparameters()
        params.update({
            'model_type': 'cnn',
            'hidden_channels': self.hidden_channels
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
                 embedding_dim: int = 128, sequence_len: int = 180) -> EmbeddingSequenceModel:
    """
    Factory function to create models.
    
    Args:
        model_type: 'cnn' or 'nn'
        days_given: number of input days
        days_predicted: number of output days
        output_type: 'bpm', 'steps', 'anxiety', or 'stress'
        embedding_dim: dimension of embeddings (default 128)
    
    Returns:
        Model instance
    """
    if model_type == 'cnn':
        return CNN(days_given, days_predicted, output_type, embedding_dim, sequence_len)
    # elif model_type == 'nn':
    #     return NNModel(days_given, days_predicted, output_type, embedding_dim)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


class MaskedBCEWithLogitsLoss(nn.Module):
    """
    Binary cross entropy loss that handles NaN values in targets.
    NaN targets are masked out and don't contribute to the loss.
    """
    
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, pred, target):
        """
        Args:
            pred: (batch, days_predicted, 1) - logits
            target: (batch, days_predicted, 1) - binary labels (may contain NaN)
        
        Returns:
            Scalar loss (averaged over valid entries only)
        """
        # Create mask for valid (non-NaN) entries
        valid_mask = ~torch.isnan(target)
        
        if valid_mask.sum() == 0:
            # No valid entries, return zero loss
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        # Replace NaN with 0 to avoid errors (will be masked anyway)
        target_clean = torch.where(valid_mask, target, torch.zeros_like(target))
        
        # Compute element-wise loss
        loss = self.bce(pred, target_clean)
        
        # Apply mask and average
        masked_loss = loss * valid_mask.float()
        return masked_loss.sum() / valid_mask.sum()


def get_loss_function(output_type: str, **kwargs):
    """
    Get appropriate loss function for output type.
    
    Args:
        output_type: 'NumberSteps', 'RestingHeartRate', 'SleepMinutesAsleep', 'anxiety', or 'stress'
        **kwargs: additional arguments (unused, for backward compatibility)
    
    Returns:
        Loss function
    """
    if output_type not in OUTPUT_TYPE_CONFIG:
        raise ValueError(f"Unknown output_type: {output_type}. Valid types: {list(OUTPUT_TYPE_CONFIG.keys())}")
    
    task_type = OUTPUT_TYPE_CONFIG[output_type]['task']
    
    if output_type in ['NumberSteps', 'RestingHeartRate', 'SleepMinutesAsleep']:
        return nn.L1Loss()
    elif task_type == 'binary':  # anxiety or stress
        return MaskedBCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown task type for output_type: {output_type}")


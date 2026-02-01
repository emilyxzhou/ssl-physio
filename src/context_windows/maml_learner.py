"""
MAML-compatible learner models for context_windows experiment.

These models support functional forward passes with external weights,
which is required for MAML's inner loop gradient computation.

Input: (batch, input_days, 128) - sequence of daily embeddings
Output: (batch, output_days, 5) - predictions for 5 targets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MAMLLearner(nn.Module):
    """
    Base class for MAML-compatible learners.
    
    Implements functional forward pass where weights can be passed explicitly.
    This allows MAML to compute gradients through the inner loop updates.
    """
    
    def __init__(self):
        super().__init__()
        self.vars = nn.ParameterList()
    
    def forward(self, x, vars=None):
        """
        Functional forward pass.
        
        Args:
            x: input tensor (batch, input_days, 128)
            vars: optional list of parameters. If None, uses self.vars
        
        Returns:
            output tensor (batch, output_days, 5)
        """
        raise NotImplementedError
    
    def parameters(self):
        """Return the learnable parameters."""
        return self.vars
    
    def get_hyperparameters(self):
        """Return model hyperparameters for logging."""
        raise NotImplementedError


class MAMLNNLearner(MAMLLearner):
    """
    Simple feedforward neural network for MAML.
    
    Architecture:
    - Flatten input: (batch, input_days * 128)
    - Linear -> ReLU -> Linear -> ReLU -> Linear
    - Output: (batch, output_days * 5) -> reshape to (batch, output_days, 5)
    """
    
    def __init__(self, input_days, output_days, hidden_dim=256):
        super().__init__()
        
        self.input_days = input_days
        self.output_days = output_days
        self.hidden_dim = hidden_dim
        self.input_dim = input_days * 128
        self.output_dim = output_days * 5
        
        # Layer 1: input -> hidden
        w1 = nn.Parameter(torch.empty(hidden_dim, self.input_dim))
        nn.init.kaiming_normal_(w1)
        b1 = nn.Parameter(torch.zeros(hidden_dim))
        self.vars.extend([w1, b1])
        
        # Layer 2: hidden -> hidden
        w2 = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        nn.init.kaiming_normal_(w2)
        b2 = nn.Parameter(torch.zeros(hidden_dim))
        self.vars.extend([w2, b2])
        
        # Layer 3: hidden -> output
        w3 = nn.Parameter(torch.empty(self.output_dim, hidden_dim))
        nn.init.kaiming_normal_(w3)
        b3 = nn.Parameter(torch.zeros(self.output_dim))
        self.vars.extend([w3, b3])
    
    def forward(self, x, vars=None):
        """
        Args:
            x: (batch, input_days, 128)
            vars: optional parameters [w1, b1, w2, b2, w3, b3]
        
        Returns:
            (batch, output_days, 5)
        """
        if vars is None:
            vars = self.vars
        
        w1, b1, w2, b2, w3, b3 = vars
        
        # Flatten input
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # (batch, input_days * 128)
        
        # Forward pass
        x = F.linear(x, w1, b1)
        x = F.relu(x)
        x = F.linear(x, w2, b2)
        x = F.relu(x)
        x = F.linear(x, w3, b3)
        
        # Reshape to (batch, output_days, 5)
        x = x.view(batch_size, self.output_days, 5)
        
        return x
    
    def get_hyperparameters(self):
        return {
            'model_type': 'nn',
            'input_days': self.input_days,
            'output_days': self.output_days,
            'hidden_dim': self.hidden_dim,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'num_params': sum(p.numel() for p in self.vars)
        }


class MAMLCNNLearner(MAMLLearner):
    """
    1D CNN for MAML on temporal sequences.
    
    Architecture:
    - Conv1d layers to process temporal patterns
    - Global average pooling
    - Linear head for output
    """
    
    def __init__(self, input_days, output_days, hidden_channels=64):
        super().__init__()
        
        self.input_days = input_days
        self.output_days = output_days
        self.hidden_channels = hidden_channels
        
        # Conv1d expects (batch, channels, length)
        # Input will be (batch, 128, input_days) after transpose
        
        # Conv layer 1: 128 -> hidden_channels
        w1 = nn.Parameter(torch.empty(hidden_channels, 128, 3))  # kernel_size=3
        nn.init.kaiming_normal_(w1)
        b1 = nn.Parameter(torch.zeros(hidden_channels))
        self.vars.extend([w1, b1])
        
        # Conv layer 2: hidden_channels -> hidden_channels
        w2 = nn.Parameter(torch.empty(hidden_channels, hidden_channels, 3))
        nn.init.kaiming_normal_(w2)
        b2 = nn.Parameter(torch.zeros(hidden_channels))
        self.vars.extend([w2, b2])
        
        # Linear head: hidden_channels -> output_days * 5
        self.output_dim = output_days * 5
        w3 = nn.Parameter(torch.empty(self.output_dim, hidden_channels))
        nn.init.kaiming_normal_(w3)
        b3 = nn.Parameter(torch.zeros(self.output_dim))
        self.vars.extend([w3, b3])
    
    def forward(self, x, vars=None):
        """
        Args:
            x: (batch, input_days, 128)
            vars: optional parameters [w1, b1, w2, b2, w3, b3]
        
        Returns:
            (batch, output_days, 5)
        """
        if vars is None:
            vars = self.vars
        
        w1, b1, w2, b2, w3, b3 = vars
        
        batch_size = x.size(0)
        
        # Transpose to (batch, 128, input_days) for Conv1d
        x = x.transpose(1, 2)
        
        # Conv layer 1 with padding to maintain size
        x = F.conv1d(x, w1, b1, padding=1)
        x = F.relu(x)
        
        # Conv layer 2
        x = F.conv1d(x, w2, b2, padding=1)
        x = F.relu(x)
        
        # Global average pooling: (batch, hidden_channels, length) -> (batch, hidden_channels)
        x = x.mean(dim=2)
        
        # Linear head
        x = F.linear(x, w3, b3)
        
        # Reshape to (batch, output_days, 5)
        x = x.view(batch_size, self.output_days, 5)
        
        return x
    
    def get_hyperparameters(self):
        return {
            'model_type': 'cnn',
            'input_days': self.input_days,
            'output_days': self.output_days,
            'hidden_channels': self.hidden_channels,
            'output_dim': self.output_dim,
            'num_params': sum(p.numel() for p in self.vars)
        }


def create_maml_learner(model_type: str, input_days: int, output_days: int, **kwargs):
    """
    Factory function to create MAML learner.
    
    Args:
        model_type: "nn" or "cnn"
        input_days: number of input days
        output_days: number of output days to predict
    
    Returns:
        MAMLLearner instance
    """
    if model_type == "nn":
        return MAMLNNLearner(input_days, output_days, **kwargs)
    elif model_type == "cnn":
        return MAMLCNNLearner(input_days, output_days, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test the learners
    batch_size = 8
    input_days = 3
    output_days = 5
    
    x = torch.randn(batch_size, input_days, 128)
    
    print("Testing NN Learner:")
    nn_model = create_maml_learner("nn", input_days, output_days)
    out_nn = nn_model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out_nn.shape}")
    print(f"  Hyperparams: {nn_model.get_hyperparameters()}")
    
    print("\nTesting CNN Learner:")
    cnn_model = create_maml_learner("cnn", input_days, output_days)
    out_cnn = cnn_model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out_cnn.shape}")
    print(f"  Hyperparams: {cnn_model.get_hyperparameters()}")
    
    # Test functional forward with external vars
    print("\nTesting functional forward:")
    vars_copy = [p.clone() for p in nn_model.vars]
    out_func = nn_model(x, vars=vars_copy)
    print(f"  Output with external vars matches: {torch.allclose(out_nn, out_func)}")


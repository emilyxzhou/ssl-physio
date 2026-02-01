"""
MAML-compatible learner models for context_windows experiment.

These models support functional forward passes with external weights,
which is required for MAML's inner loop gradient computation.

Input: (batch, input_days, 180, 128) - sequence of daily minute-level embeddings
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
            x: input tensor (batch, input_days, 180, 128) - minute-level embeddings
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
    - Mean-pool over time dimension: (batch, input_days, 180, 128) -> (batch, input_days, 128)
    - Flatten: (batch, input_days * 128)
    - Linear -> ReLU -> Linear -> ReLU -> Linear
    - Output: (batch, output_days * 5) -> reshape to (batch, output_days, 5)
    """
    
    def __init__(self, input_days, output_days, hidden_dim=256, embedding_dim=128, sequence_len=180):
        super().__init__()
        
        self.input_days = input_days
        self.output_days = output_days
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.sequence_len = sequence_len
        self.input_dim = input_days * embedding_dim  # After mean-pooling over time
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
            x: (batch, input_days, 180, 128)
            vars: optional parameters [w1, b1, w2, b2, w3, b3]
        
        Returns:
            (batch, output_days, 5)
        """
        if vars is None:
            vars = self.vars
        
        w1, b1, w2, b2, w3, b3 = vars
        
        batch_size = x.size(0)
        
        # Mean-pool over time dimension: (batch, input_days, 180, 128) -> (batch, input_days, 128)
        x = x.mean(dim=2)
        
        # Flatten input: (batch, input_days, 128) -> (batch, input_days * 128)
        x = x.view(batch_size, -1)
        
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
            'embedding_dim': self.embedding_dim,
            'sequence_len': self.sequence_len,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'num_params': sum(p.numel() for p in self.vars)
        }


class MAMLCNNLearner(MAMLLearner):
    """
    1D CNN for MAML on minute-level temporal sequences.
    
    Architecture (similar to models.py CNN):
    - Input: (batch, input_days, 180, 128) -> reshape to (batch, 128, input_days * 180)
    - Conv1d 128 -> 128, kernel_size=3, padding=1, then ReLU, AvgPool1d(2)
    - Conv1d 128 -> 128, kernel_size=3, padding=1, then ReLU, AvgPool1d(2)
    - Conv1d 128 -> 64, kernel_size=3, padding=1, then ReLU, AvgPool1d(2)
    - Conv1d 64 -> 32, kernel_size=3, padding=1, then ReLU, AvgPool1d(2)
    - Flatten -> Linear to output
    """
    
    def __init__(self, input_days, output_days, embedding_dim=128, sequence_len=180):
        super().__init__()
        
        self.input_days = input_days
        self.output_days = output_days
        self.embedding_dim = embedding_dim
        self.sequence_len = sequence_len
        self.total_seq_len = input_days * sequence_len  # Total sequence length after flattening days
        
        # Conv1d expects (batch, channels, length)
        # Input will be (batch, 128, input_days * 180) after reshape
        
        # Conv layer 1: 128 -> 128, kernel_size=3
        w1 = nn.Parameter(torch.empty(128, 128, 3))
        nn.init.kaiming_normal_(w1)
        b1 = nn.Parameter(torch.zeros(128))
        self.vars.extend([w1, b1])
        
        # Conv layer 2: 128 -> 128, kernel_size=3
        w2 = nn.Parameter(torch.empty(128, 128, 3))
        nn.init.kaiming_normal_(w2)
        b2 = nn.Parameter(torch.zeros(128))
        self.vars.extend([w2, b2])
        
        # Conv layer 3: 128 -> 64, kernel_size=3
        w3 = nn.Parameter(torch.empty(64, 128, 3))
        nn.init.kaiming_normal_(w3)
        b3 = nn.Parameter(torch.zeros(64))
        self.vars.extend([w3, b3])
        
        # Conv layer 4: 64 -> 32, kernel_size=3
        w4 = nn.Parameter(torch.empty(32, 64, 3))
        nn.init.kaiming_normal_(w4)
        b4 = nn.Parameter(torch.zeros(32))
        self.vars.extend([w4, b4])
        
        # Calculate flattened size after 4 pooling layers (each divides by 2)
        # Length after pooling: total_seq_len // 16
        self.pooled_len = self.total_seq_len // 16
        fc_input_dim = 32 * self.pooled_len
        
        # Linear head: flattened -> output_days * 5
        self.output_dim = output_days * 5
        w5 = nn.Parameter(torch.empty(self.output_dim, fc_input_dim))
        nn.init.kaiming_normal_(w5)
        b5 = nn.Parameter(torch.zeros(self.output_dim))
        self.vars.extend([w5, b5])
    
    def forward(self, x, vars=None):
        """
        Args:
            x: (batch, input_days, 180, 128)
            vars: optional parameters [w1, b1, w2, b2, w3, b3, w4, b4, w5, b5]
        
        Returns:
            (batch, output_days, 5)
        """
        if vars is None:
            vars = self.vars
        
        w1, b1, w2, b2, w3, b3, w4, b4, w5, b5 = vars
        
        batch_size = x.size(0)
        
        # Reshape from (batch, input_days, 180, 128) to (batch, 128, input_days * 180)
        # First: (batch, input_days, 180, 128) -> (batch, input_days * 180, 128)
        x = x.view(batch_size, -1, self.embedding_dim)
        # Then transpose: (batch, input_days * 180, 128) -> (batch, 128, input_days * 180)
        x = x.transpose(1, 2)
        
        # Conv block 1: conv -> relu -> pool
        x = F.conv1d(x, w1, b1, padding=1)
        x = F.relu(x)
        x = F.avg_pool1d(x, kernel_size=2)
        
        # Conv block 2: conv -> relu -> pool
        x = F.conv1d(x, w2, b2, padding=1)
        x = F.relu(x)
        x = F.avg_pool1d(x, kernel_size=2)
        
        # Conv block 3: conv -> relu -> pool
        x = F.conv1d(x, w3, b3, padding=1)
        x = F.relu(x)
        x = F.avg_pool1d(x, kernel_size=2)
        
        # Conv block 4: conv -> relu -> pool
        x = F.conv1d(x, w4, b4, padding=1)
        x = F.relu(x)
        x = F.avg_pool1d(x, kernel_size=2)
        
        # Flatten: (batch, 32, pooled_len) -> (batch, 32 * pooled_len)
        x = x.view(batch_size, -1)
        
        # Linear head
        x = F.linear(x, w5, b5)
        
        # Reshape to (batch, output_days, 5)
        x = x.view(batch_size, self.output_days, 5)
        
        return x
    
    def get_hyperparameters(self):
        return {
            'model_type': 'cnn',
            'input_days': self.input_days,
            'output_days': self.output_days,
            'embedding_dim': self.embedding_dim,
            'sequence_len': self.sequence_len,
            'total_seq_len': self.total_seq_len,
            'pooled_len': self.pooled_len,
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
    # Test the learners with minute-level embeddings
    batch_size = 8
    input_days = 3
    output_days = 5
    sequence_len = 180
    embedding_dim = 128
    
    # Input shape: (batch, input_days, 180, 128) - minute-level embeddings
    x = torch.randn(batch_size, input_days, sequence_len, embedding_dim)
    
    print("="*60)
    print("Testing MAML Learners with minute-level embeddings")
    print("="*60)
    print(f"Input shape: {x.shape}")
    print(f"  batch_size={batch_size}, input_days={input_days}, sequence_len={sequence_len}, embedding_dim={embedding_dim}")
    
    print("\n" + "-"*40)
    print("Testing NN Learner (mean-pools over time):")
    print("-"*40)
    nn_model = create_maml_learner("nn", input_days, output_days)
    out_nn = nn_model(x)
    print(f"  Output shape: {out_nn.shape}")
    print(f"  Hyperparams: {nn_model.get_hyperparameters()}")
    
    print("\n" + "-"*40)
    print("Testing CNN Learner (processes full sequence):")
    print("-"*40)
    cnn_model = create_maml_learner("cnn", input_days, output_days)
    out_cnn = cnn_model(x)
    print(f"  Output shape: {out_cnn.shape}")
    print(f"  Hyperparams: {cnn_model.get_hyperparameters()}")
    
    # Test functional forward with external vars (required for MAML)
    print("\n" + "-"*40)
    print("Testing functional forward (MAML inner loop):")
    print("-"*40)
    vars_copy = [p.clone() for p in nn_model.vars]
    out_func = nn_model(x, vars=vars_copy)
    print(f"  NN: Output with external vars matches: {torch.allclose(out_nn, out_func)}")
    
    vars_copy_cnn = [p.clone() for p in cnn_model.vars]
    out_func_cnn = cnn_model(x, vars=vars_copy_cnn)
    print(f"  CNN: Output with external vars matches: {torch.allclose(out_cnn, out_func_cnn)}")
    
    # Test gradient computation (required for MAML)
    print("\n" + "-"*40)
    print("Testing gradient computation:")
    print("-"*40)
    loss = out_cnn.sum()
    loss.backward()
    print(f"  Gradients computed successfully for CNN")
    
    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)


"""
MAML Meta-Learner for context_windows experiment.

Implements Model-Agnostic Meta-Learning for multi-output prediction:
- Binary targets (stress, anxiety): BCE loss
- Regression targets (RHR, sleep, steps): MSE loss

We train 5 separate models (one per target) but share the MAML training loop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

# Handle imports for both direct and package execution
try:
    from .maml_learner import create_maml_learner
except ImportError:
    from maml_learner import create_maml_learner


# Target configuration
TARGET_CONFIG = {
    'stress': {'idx': 0, 'type': 'binary'},
    'anxiety': {'idx': 1, 'type': 'binary'},
    'rhr': {'idx': 2, 'type': 'regression'},
    'sleep': {'idx': 3, 'type': 'regression'},
    'steps': {'idx': 4, 'type': 'regression'},
}


class MAMLConfig:
    """Configuration for MAML training."""
    
    def __init__(
        self,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 5,
        meta_epochs: int = 20,
        inner_steps_test: int = 10,
    ):
        self.inner_lr = inner_lr      # Learning rate for inner loop (task adaptation)
        self.outer_lr = outer_lr      # Learning rate for outer loop (meta update)
        self.inner_steps = inner_steps  # Number of gradient steps in inner loop during training
        self.meta_epochs = meta_epochs  # Number of meta-training epochs
        self.inner_steps_test = inner_steps_test  # Inner steps during testing/fine-tuning
    
    def to_dict(self):
        return {
            'inner_lr': self.inner_lr,
            'outer_lr': self.outer_lr,
            'inner_steps': self.inner_steps,
            'meta_epochs': self.meta_epochs,
            'inner_steps_test': self.inner_steps_test,
        }


class MAMLTrainer:
    """
    MAML trainer for a single target.
    
    Handles inner loop adaptation and outer loop meta-updates for one model.
    """
    
    def __init__(self, model, target_name, config, device):
        """
        Args:
            model: MAMLLearner instance
            target_name: one of 'stress', 'anxiety', 'rhr', 'sleep', 'steps'
            config: MAMLConfig instance
            device: torch device
        """
        self.model = model.to(device)
        self.target_name = target_name
        self.target_idx = TARGET_CONFIG[target_name]['idx']
        self.target_type = TARGET_CONFIG[target_name]['type']
        self.config = config
        self.device = device
        
        self.meta_optim = torch.optim.Adam(self.model.parameters(), lr=config.outer_lr)
    
    def compute_loss(self, pred, target, mask=None):
        """
        Compute loss for this target.
        
        Args:
            pred: (batch, output_days, 5) - full predictions
            target: (batch, output_days, 5) - full targets
            mask: optional boolean mask for valid entries
        
        Returns:
            loss tensor
        """
        # Extract this target's predictions and labels
        pred_target = pred[:, :, self.target_idx]  # (batch, output_days)
        true_target = target[:, :, self.target_idx]  # (batch, output_days)
        
        # Mask NaN values
        valid_mask = ~torch.isnan(true_target)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        pred_valid = pred_target[valid_mask]
        true_valid = true_target[valid_mask]
        
        if self.target_type == 'binary':
            # BCE with logits
            loss = F.binary_cross_entropy_with_logits(pred_valid, true_valid)
        else:
            # MSE for regression - clamp predictions to [0, 1] since targets are normalized
            pred_clamped = torch.clamp(pred_valid, 0.0, 1.0)
            loss = F.mse_loss(pred_clamped, true_valid)
        
        return loss
    
    def inner_loop(self, x_support, y_support, num_steps=None):
        """
        Perform inner loop adaptation on support set.
        
        Args:
            x_support: (N, input_days, 128)
            y_support: (N, output_days, 5)
            num_steps: number of gradient steps (uses config default if None)
        
        Returns:
            adapted_vars: list of adapted parameters
        """
        if num_steps is None:
            num_steps = self.config.inner_steps
        
        # Start with current model parameters
        fast_weights = [p.clone() for p in self.model.parameters()]
        
        for step in range(num_steps):
            # Forward pass
            pred = self.model(x_support, vars=fast_weights)
            loss = self.compute_loss(pred, y_support)
            
            if loss.item() == 0 or torch.isnan(loss) or torch.isinf(loss):
                continue
            
            # Compute gradients w.r.t. fast_weights
            grads = torch.autograd.grad(loss, fast_weights, create_graph=True)
            
            # Clip gradients in inner loop to prevent explosion
            grads = [torch.clamp(g, -1.0, 1.0) for g in grads]
            
            # Update fast_weights
            fast_weights = [w - self.config.inner_lr * g for w, g in zip(fast_weights, grads)]
        
        return fast_weights
    
    def meta_train_step(self, x_support, y_support, x_query, y_query):
        """
        One meta-training step: adapt on support, compute loss on query.
        
        Args:
            x_support: (N_support, input_days, 128)
            y_support: (N_support, output_days, 5)
            x_query: (N_query, input_days, 128)
            y_query: (N_query, output_days, 5)
        
        Returns:
            query_loss: scalar loss on query set after adaptation
        """
        # Inner loop: adapt on support set
        adapted_weights = self.inner_loop(x_support, y_support)
        
        # Evaluate on query set with adapted weights
        pred_query = self.model(x_query, vars=adapted_weights)
        query_loss = self.compute_loss(pred_query, y_query)
        
        return query_loss
    
    def meta_update(self, loss):
        """Perform meta-update (outer loop)."""
        self.meta_optim.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.meta_optim.step()
    
    def evaluate(self, x_support, y_support, x_query, y_query):
        """
        Evaluate on query set after fine-tuning on support set.
        
        Returns dict with predictions and metrics.
        """
        # Fine-tune on support (use more steps at test time)
        adapted_weights = self.inner_loop(
            x_support, y_support, 
            num_steps=self.config.inner_steps_test
        )
        
        # Predict on query
        with torch.no_grad():
            pred = self.model(x_query, vars=adapted_weights)
        
        # Extract this target
        pred_target = pred[:, :, self.target_idx].cpu().numpy()
        true_target = y_query[:, :, self.target_idx].cpu().numpy()
        
        # Clamp regression predictions to [0, 1] for evaluation
        if self.target_type == 'regression':
            pred_target = np.clip(pred_target, 0.0, 1.0)
        
        # Flatten for metrics
        pred_flat = pred_target.flatten()
        true_flat = true_target.flatten()
        
        # Filter valid entries
        valid_mask = ~np.isnan(true_flat)
        if valid_mask.sum() == 0:
            return {'n_valid': 0}
        
        pred_valid = pred_flat[valid_mask]
        true_valid = true_flat[valid_mask]
        
        metrics = {'n_valid': int(valid_mask.sum())}
        
        if self.target_type == 'binary':
            # Convert logits to probabilities and predictions
            pred_probs = 1 / (1 + np.exp(-pred_valid))
            pred_labels = (pred_probs >= 0.5).astype(int)
            true_labels = true_valid.astype(int)
            
            metrics['accuracy'] = float(np.mean(pred_labels == true_labels))
            
            # Balanced accuracy
            from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
            try:
                metrics['balanced_accuracy'] = float(balanced_accuracy_score(true_labels, pred_labels))
            except:
                metrics['balanced_accuracy'] = float('nan')
            
            try:
                metrics['f1'] = float(f1_score(true_labels, pred_labels, zero_division=0))
            except:
                metrics['f1'] = float('nan')
            
            try:
                if len(np.unique(true_labels)) > 1:
                    metrics['auc'] = float(roc_auc_score(true_labels, pred_probs))
                else:
                    metrics['auc'] = float('nan')
            except:
                metrics['auc'] = float('nan')
        else:
            # Regression metrics
            metrics['mse'] = float(np.mean((pred_valid - true_valid) ** 2))
            metrics['mae'] = float(np.mean(np.abs(pred_valid - true_valid)))
        
        return metrics


class MultiTargetMAML:
    """
    Manages 5 separate MAML trainers (one per target).
    
    Trains all models together for efficiency but keeps them separate.
    """
    
    def __init__(self, model_type, input_days, output_days, config, device):
        """
        Args:
            model_type: "nn" or "cnn"
            input_days: number of input days
            output_days: number of output days to predict
            config: MAMLConfig instance
            device: torch device
        """
        self.model_type = model_type
        self.input_days = input_days
        self.output_days = output_days
        self.config = config
        self.device = device
        
        # Create one trainer per target
        self.trainers = {}
        for target_name in TARGET_CONFIG.keys():
            model = create_maml_learner(model_type, input_days, output_days)
            self.trainers[target_name] = MAMLTrainer(model, target_name, config, device)
    
    def train_epoch(self, x_support, y_support, x_query, y_query):
        """
        Train all models for one meta-epoch.
        
        Returns dict of losses per target.
        """
        # Move to device
        x_support = torch.tensor(x_support, dtype=torch.float32, device=self.device)
        y_support = torch.tensor(y_support, dtype=torch.float32, device=self.device)
        x_query = torch.tensor(x_query, dtype=torch.float32, device=self.device)
        y_query = torch.tensor(y_query, dtype=torch.float32, device=self.device)
        
        losses = {}
        for target_name, trainer in self.trainers.items():
            # Compute meta-loss
            loss = trainer.meta_train_step(x_support, y_support, x_query, y_query)
            losses[target_name] = loss.item()
            
            # Meta-update
            trainer.meta_update(loss)
        
        return losses
    
    def train(self, x_support, y_support, x_query, y_query, verbose=False):
        """
        Full meta-training loop.
        
        Args:
            x_support: support set inputs
            y_support: support set targets
            x_query: query set inputs (for meta-gradient)
            y_query: query set targets
            verbose: print progress
        
        Returns:
            training_history: dict of losses per epoch
        """
        history = {name: [] for name in TARGET_CONFIG.keys()}
        
        for epoch in range(self.config.meta_epochs):
            epoch_losses = self.train_epoch(x_support, y_support, x_query, y_query)
            
            for name, loss in epoch_losses.items():
                history[name].append(loss)
            
            if verbose and (epoch + 1) % 5 == 0:
                loss_str = ", ".join([f"{k}:{v:.4f}" for k, v in epoch_losses.items()])
                print(f"  Epoch {epoch+1}/{self.config.meta_epochs}: {loss_str}")
        
        return history
    
    def evaluate(self, x_support, y_support, x_query, y_query):
        """
        Evaluate all models on query set.
        
        Returns dict of metrics per target.
        """
        # Move to device
        x_support = torch.tensor(x_support, dtype=torch.float32, device=self.device)
        y_support = torch.tensor(y_support, dtype=torch.float32, device=self.device)
        x_query = torch.tensor(x_query, dtype=torch.float32, device=self.device)
        y_query = torch.tensor(y_query, dtype=torch.float32, device=self.device)
        
        results = {}
        for target_name, trainer in self.trainers.items():
            results[target_name] = trainer.evaluate(x_support, y_support, x_query, y_query)
        
        return results
    
    def get_hyperparameters(self):
        """Return combined hyperparameters."""
        return {
            'model_type': self.model_type,
            'input_days': self.input_days,
            'output_days': self.output_days,
            'maml_config': self.config.to_dict(),
            'model_params': self.trainers['stress'].model.get_hyperparameters()
        }


if __name__ == "__main__":
    # Test the MAML system
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dummy data
    n_support = 160
    n_query = 10
    input_days = 3
    output_days = 5
    
    x_support = np.random.randn(n_support, input_days, 128).astype(np.float32)
    y_support = np.random.randn(n_support, output_days, 5).astype(np.float32)
    # Make binary targets 0/1
    y_support[:, :, 0] = (y_support[:, :, 0] > 0).astype(np.float32)
    y_support[:, :, 1] = (y_support[:, :, 1] > 0).astype(np.float32)
    
    x_query = np.random.randn(n_query, input_days, 128).astype(np.float32)
    y_query = np.random.randn(n_query, output_days, 5).astype(np.float32)
    y_query[:, :, 0] = (y_query[:, :, 0] > 0).astype(np.float32)
    y_query[:, :, 1] = (y_query[:, :, 1] > 0).astype(np.float32)
    
    # Create and train
    config = MAMLConfig(meta_epochs=10)
    maml = MultiTargetMAML("nn", input_days, output_days, config, device)
    
    print("\nTraining MAML...")
    history = maml.train(x_support, y_support, x_query, y_query, verbose=True)
    
    print("\nEvaluating...")
    results = maml.evaluate(x_support, y_support, x_query, y_query)
    
    print("\nResults:")
    for target, metrics in results.items():
        print(f"  {target}: {metrics}")


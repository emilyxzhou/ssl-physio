"""
Context Windows MAML Experiment Module

MAML-based within-subject prediction using pre-computed embeddings.
"""

from .context_windows import run_context_windows
from .maml_meta import MAMLConfig, MultiTargetMAML
from .maml_learner import create_maml_learner, MAMLNNLearner, MAMLCNNLearner
from .data_loader import load_all_data, sample_support_windows, create_query_windows

__all__ = [
    'run_context_windows',
    'MAMLConfig',
    'MultiTargetMAML',
    'create_maml_learner',
    'MAMLNNLearner',
    'MAMLCNNLearner',
    'load_all_data',
    'sample_support_windows',
    'create_query_windows',
]


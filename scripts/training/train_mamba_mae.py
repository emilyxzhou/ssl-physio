import os
import sys
from pathlib import Path
USER_ROOT = str(Path(__file__).resolve().parents[3])
paths = [
    os.path.join(
        USER_ROOT, "ssl-physio", "src", "dataloaders"
    ),
    os.path.join(
        USER_ROOT, "ssl-physio", "src", "mamba"
    ),
    os.path.join(
        USER_ROOT, "ssl-physio", "src", "trainers"
    )
]
for path in paths:
    sys.path.insert(0, path)

import argparse
import logging
import math
import numpy as np
import pprint
import random
import signal
import time
import torch
import torch.nn as nn
import wandb
import yaml

from datetime import datetime
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

from trainer import Trainer
from tiles_dataloader import load_tiles_open, load_tiles_holdout, TilesDataset

# import mamba models
# from mamba import ---

# Define logging console
import logging
logging.basicConfig(
    format="%(asctime)s %(levelname)-3s ==> %(message)s", 
    level=logging.INFO, 
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Save paths 
MODEL_SAVE_FOLDER = f"{USER_ROOT}/ssl-physio/models/reconstruction"

CHECKPOINT_DIR = f"{USER_ROOT}/ssl-physio/ckpts"
CHECKPOINT_PREFIX = "mamba"
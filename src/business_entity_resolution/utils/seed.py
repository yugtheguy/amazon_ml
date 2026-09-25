import os
import random
import numpy as np

def seed_everything(seed: int = 42):
    """Sets seed for random, numpy, and python hash seed for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # If PyTorch is introduced later, add torch.manual_seed(seed)

"""
PhysioGraph: Spatio-Temporal Graph Neural Network with Multi-Horizon Auxiliary Supervision
Package initialization.
"""

from . import data
from . import models
from . import train
from . import evaluation

__version__ = "3.0.0"
__all__ = ["data", "models", "train", "evaluation"]

"""
Federated learning and DP-SGD training harness for Track B.
"""

from .centralized_baseline import run_centralized_training
from .client import ClinicalFlowerClient
from .server import run_federated_simulation

__all__ = [
    "run_centralized_training",
    "ClinicalFlowerClient",
    "run_federated_simulation",
]

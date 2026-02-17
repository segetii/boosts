"""
Unified Observation Framework

A mathematically grounded, non-ML anomaly and mimic detection system that combines
multiple mathematical domains into a single unified action score.

Integrates:
- Probability / KL Divergence
- Chaos / Lyapunov Exponent
- Frequency / Spectral Divergence
- Geometry / Hellinger Distance
"""

__version__ = "0.1.0"

from . import metrics
from . import weights
from . import scoring
from . import classification
from . import clustering
from . import utils

__all__ = [
    "metrics",
    "weights",
    "scoring",
    "classification",
    "clustering",
    "utils",
]

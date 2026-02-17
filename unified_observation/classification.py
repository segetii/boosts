"""
Classification methods for anomaly detection.

Implements multiple thresholding strategies to classify observations.
"""

import numpy as np
from typing import Tuple
from kneed import KneeLocator
from .utils import ensure_float64


def classify(
    log_scores: np.ndarray,
    method: str = 'percentile',
    threshold_param: float = 95
) -> Tuple[np.ndarray, float]:
    """
    Classify observations as normal (0) or anomaly (1).
    
    Args:
        log_scores: Array of log-action scores
        method: One of 'percentile', 'sigma', 'knee'
        threshold_param: Parameter for the method:
            - percentile: percentile value (e.g., 95 for 95th percentile)
            - sigma: number of standard deviations (e.g., 3 for 3-sigma)
            - knee: not used (auto-detected)
            
    Returns:
        Tuple of (labels, threshold):
            - labels: Boolean array (True = anomaly, False = normal)
            - threshold: The threshold value used
    """
    log_scores = ensure_float64(log_scores)
    
    if method == 'percentile':
        threshold = np.percentile(log_scores, threshold_param)
        labels = log_scores > threshold
        
    elif method == 'sigma':
        mean = np.mean(log_scores)
        std = np.std(log_scores)
        threshold = mean + threshold_param * std
        labels = log_scores > threshold
        
    elif method == 'knee':
        # Use kneedle algorithm to find natural threshold
        sorted_scores = np.sort(log_scores)
        x = np.arange(len(sorted_scores))
        
        # Find knee point
        try:
            kneedle = KneeLocator(
                x, sorted_scores,
                curve='convex',
                direction='increasing',
                S=1.0
            )
            
            if kneedle.knee is not None:
                threshold = sorted_scores[kneedle.knee]
            else:
                # Fallback to 95th percentile if knee not found
                threshold = np.percentile(log_scores, 95)
        except Exception:
            # Fallback to 95th percentile on any error
            threshold = np.percentile(log_scores, 95)
        
        labels = log_scores > threshold
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return labels.astype(bool), float(threshold)

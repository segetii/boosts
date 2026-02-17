"""
Utility functions for the Unified Observation Framework.

Provides memory estimation, safe math helpers, and data validation.
"""

import numpy as np
from typing import Union, Tuple


def safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Safe logarithm with epsilon protection.
    
    Args:
        x: Input array
        eps: Epsilon value to prevent log(0)
        
    Returns:
        Log of (x + eps)
    """
    return np.log(np.maximum(x, eps) + eps)


def safe_divide(numerator: np.ndarray, denominator: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Safe division with epsilon protection.
    
    Args:
        numerator: Numerator array
        denominator: Denominator array
        eps: Epsilon value to prevent division by zero
        
    Returns:
        numerator / (denominator + eps)
    """
    return numerator / (np.abs(denominator) + eps)


def validate_data(X: np.ndarray, name: str = "data") -> None:
    """
    Validate input data for NaN, inf, and other issues.
    
    Args:
        X: Input data array
        name: Name of the data for error messages
        
    Raises:
        ValueError: If data contains NaN or inf
        TypeError: If data is not a numpy array or cannot be validated
    """
    if not isinstance(X, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    
    if X.dtype not in [np.float32, np.float64]:
        raise TypeError(f"{name} must be float32 or float64, got {X.dtype}. Convert before validation.")
    
    if np.any(np.isnan(X)):
        raise ValueError(f"{name} contains NaN values")
    
    if np.any(np.isinf(X)):
        raise ValueError(f"{name} contains infinite values")


def estimate_memory(N: int, m: int) -> dict:
    """
    Estimate memory requirements for given data dimensions.
    
    Args:
        N: Number of observations
        m: Number of features per observation
        
    Returns:
        Dictionary with memory estimates in GB
    """
    bytes_per_float64 = 8
    
    raw_data = (N * m * bytes_per_float64) / (1024**3)
    reference = (m * bytes_per_float64) / (1024**3)
    scores = (N * 5 * bytes_per_float64) / (1024**3)  # 4 metrics + 1 unified
    fft_buffer = (m * bytes_per_float64 * 2) / (1024**3)  # Complex FFT
    
    streaming_total = raw_data + reference + scores + fft_buffer
    pairwise_full = (N * N * bytes_per_float64) / (1024**3)
    
    return {
        "raw_data_gb": raw_data,
        "reference_gb": reference,
        "scores_gb": scores,
        "fft_buffer_gb": fft_buffer,
        "streaming_total_gb": streaming_total,
        "pairwise_full_gb": pairwise_full,
        "safe_for_50gb": streaming_total < 50.0,
    }


def ensure_float64(X: np.ndarray) -> np.ndarray:
    """
    Ensure array is float64 for maximum numerical range.
    
    Args:
        X: Input array
        
    Returns:
        Array converted to float64
    """
    if X.dtype != np.float64:
        return X.astype(np.float64)
    return X


def handle_negative_values(x: np.ndarray) -> np.ndarray:
    """
    Handle negative values by shifting to make all values non-negative.
    
    Args:
        x: Input array
        
    Returns:
        Array with all non-negative values
    """
    min_val = np.min(x)
    if min_val < 0:
        return x - min_val
    return x


def compute_robust_statistics(X: np.ndarray) -> dict:
    """
    Compute robust statistics for an array.
    
    Args:
        X: Input array (N, m) or (m,)
        
    Returns:
        Dictionary with statistics
    """
    return {
        "mean": np.mean(X, axis=0) if X.ndim > 1 else np.mean(X),
        "median": np.median(X, axis=0) if X.ndim > 1 else np.median(X),
        "std": np.std(X, axis=0) if X.ndim > 1 else np.std(X),
        "min": np.min(X, axis=0) if X.ndim > 1 else np.min(X),
        "max": np.max(X, axis=0) if X.ndim > 1 else np.max(X),
        "q25": np.percentile(X, 25, axis=0) if X.ndim > 1 else np.percentile(X, 25),
        "q75": np.percentile(X, 75, axis=0) if X.ndim > 1 else np.percentile(X, 75),
    }


def cap_iterations(current_iter: int, max_iter: int, loop_name: str = "loop") -> bool:
    """
    Check if iterations should be capped.
    
    Args:
        current_iter: Current iteration number
        max_iter: Maximum allowed iterations
        loop_name: Name of the loop for warning messages
        
    Returns:
        True if should continue, False if should stop
    """
    if current_iter >= max_iter:
        import warnings
        warnings.warn(f"{loop_name} reached maximum iterations ({max_iter})")
        return False
    return True

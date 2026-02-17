"""
Core metric functions for the Unified Observation Framework.

Implements:
- KL Divergence (probability)
- Lyapunov Exponent (chaos)
- Spectral Divergence (frequency)
- Hellinger Distance (geometry)
- Probability normalization
"""

import numpy as np
from typing import Union
from .utils import safe_log, safe_divide, handle_negative_values, ensure_float64


def normalize_to_probability(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Convert a raw feature vector to a probability distribution.
    
    Handles negative values by shifting, then normalizes and applies epsilon smoothing.
    
    Args:
        x: Input array (can contain negative values)
        eps: Epsilon for smoothing
        
    Returns:
        Probability distribution (sums to 1)
    """
    x = ensure_float64(x)
    
    # Handle negative values
    x_shifted = handle_negative_values(x)
    
    # Normalize to sum to 1
    sum_x = np.sum(x_shifted)
    if sum_x == 0:
        # Uniform distribution if all zeros
        p = np.ones_like(x_shifted) / len(x_shifted)
    else:
        p = x_shifted / sum_x
    
    # Apply epsilon smoothing
    p = p + eps
    # Re-normalize after smoothing
    p = p / np.sum(p)
    
    return p


def kl_divergence(p: np.ndarray, p_ref: np.ndarray, eps: float = 1e-12) -> float:
    """
    Compute KL divergence between two probability distributions.
    
    Both distributions are epsilon-smoothed and re-normalized before computing.
    
    Args:
        p: First probability distribution
        p_ref: Reference probability distribution
        eps: Epsilon for smoothing and numerical safety
        
    Returns:
        KL divergence value (in nats, not bits)
    """
    p = ensure_float64(p)
    p_ref = ensure_float64(p_ref)
    
    # Ensure same shape
    if p.shape != p_ref.shape:
        raise ValueError(f"Shape mismatch: p {p.shape} vs p_ref {p_ref.shape}")
    
    # Epsilon smoothing and re-normalization
    p_smooth = p + eps
    p_smooth = p_smooth / np.sum(p_smooth)
    
    p_ref_smooth = p_ref + eps
    p_ref_smooth = p_ref_smooth / np.sum(p_ref_smooth)
    
    # Compute KL divergence: sum(p * log(p / p_ref))
    # Using log properties: log(p/p_ref) = log(p) - log(p_ref)
    kl = np.sum(p_smooth * (safe_log(p_smooth, eps) - safe_log(p_ref_smooth, eps)))
    
    return float(kl)


def lyapunov_exponent(f: np.ndarray, f_ref: np.ndarray, eps: float = 1e-12) -> float:
    """
    Compute pseudo-Lyapunov exponent measuring dynamic divergence.
    
    Formula:
        λ = (1/(m-1)) * Σ log(|f(t_{j+1}) - f_ref(t_{j+1})| / (|f(t_j) - f_ref(t_j)| + eps))
    
    Args:
        f: Signal array
        f_ref: Reference signal array
        eps: Epsilon for numerical safety (division and log protection)
        
    Returns:
        Pseudo-Lyapunov exponent
    """
    f = ensure_float64(f)
    f_ref = ensure_float64(f_ref)
    
    if f.shape != f_ref.shape:
        raise ValueError(f"Shape mismatch: f {f.shape} vs f_ref {f_ref.shape}")
    
    if len(f) < 2:
        return 0.0
    
    # Compute differences at each time point
    diff = np.abs(f - f_ref)
    
    # Compute ratios of consecutive differences
    # ratio = |diff[j+1]| / (|diff[j]| + eps)
    diff_j = diff[:-1]
    diff_j_plus_1 = diff[1:]
    
    # Safe division with epsilon in denominator
    ratios = safe_divide(diff_j_plus_1, diff_j, eps)
    
    # Add epsilon inside log to prevent log(0)
    log_ratios = safe_log(ratios, eps)
    
    # Average over all time steps
    m = len(f)
    lyapunov = np.sum(log_ratios) / (m - 1)
    
    return float(lyapunov)


def spectral_divergence(f: np.ndarray, f_ref: np.ndarray) -> float:
    """
    Compute spectral divergence using FFT.
    
    Returns sum of squared differences of Fourier coefficients:
        D_freq = Σ |F_i(k) - F_ref(k)|²
    
    Note: Computes FFT on-the-fly and discards to save memory.
    
    Args:
        f: Signal array
        f_ref: Reference signal array
        
    Returns:
        Spectral divergence value
    """
    f = ensure_float64(f)
    f_ref = ensure_float64(f_ref)
    
    if f.shape != f_ref.shape:
        raise ValueError(f"Shape mismatch: f {f.shape} vs f_ref {f_ref.shape}")
    
    # Compute FFT (returns complex array)
    F = np.fft.fft(f)
    F_ref = np.fft.fft(f_ref)
    
    # Compute squared differences of coefficients
    diff = F - F_ref
    spectral_div = np.sum(np.abs(diff)**2)
    
    # No need to store FFTs - they are garbage collected after this function
    
    return float(spectral_div)


def hellinger_distance(p: np.ndarray, p_ref: np.ndarray) -> float:
    """
    Compute Hellinger distance as a proxy for geodesic manifold distance.
    
    Formula:
        d_H = (1/√2) * √(Σ (√p_i - √p_ref_i)²)
    
    Args:
        p: First probability distribution
        p_ref: Reference probability distribution
        
    Returns:
        Hellinger distance
    """
    p = ensure_float64(p)
    p_ref = ensure_float64(p_ref)
    
    if p.shape != p_ref.shape:
        raise ValueError(f"Shape mismatch: p {p.shape} vs p_ref {p_ref.shape}")
    
    # Take square roots
    sqrt_p = np.sqrt(np.maximum(p, 0))  # Ensure non-negative before sqrt
    sqrt_p_ref = np.sqrt(np.maximum(p_ref, 0))
    
    # Compute squared differences and sum
    squared_diff = (sqrt_p - sqrt_p_ref)**2
    sum_squared_diff = np.sum(squared_diff)
    
    # Hellinger distance
    hellinger = (1.0 / np.sqrt(2.0)) * np.sqrt(sum_squared_diff)
    
    return float(hellinger)

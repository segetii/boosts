"""
Unified action score computation with streaming processing.

Implements the main entry point for computing observation scores.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from .metrics import (
    kl_divergence,
    lyapunov_exponent,
    spectral_divergence,
    hellinger_distance,
    normalize_to_probability,
)
from .weights import (
    variance_ratio_weights,
    iqr_weights,
    coefficient_of_variation_weights,
    mutual_information_weights,
    fisher_weights,
    combined_weights,
)
from .utils import ensure_float64, validate_data, cap_iterations


def compute_reference(
    X: np.ndarray,
    method: str = 'trimmed_mean',
    trim_fraction: float = 0.1,
    max_iterations: int = 10
) -> np.ndarray:
    """
    Compute reference signal from data.
    
    Args:
        X: Input data array of shape (N, m)
        method: One of 'centroid', 'median', 'trimmed_mean'
        trim_fraction: Fraction of outliers to trim (for trimmed_mean)
        max_iterations: Maximum iterations for iterative trimming
        
    Returns:
        Reference signal of shape (m,)
    """
    X = ensure_float64(X)
    validate_data(X, "X")
    
    if method == 'centroid':
        return np.mean(X, axis=0)
    
    elif method == 'median':
        return np.median(X, axis=0)
    
    elif method == 'trimmed_mean':
        # Iteratively trim outliers and recompute mean
        current_X = X.copy()
        
        for iteration in range(max_iterations):
            if not cap_iterations(iteration, max_iterations, "trimmed_mean"):
                break
            
            # Compute current mean
            current_mean = np.mean(current_X, axis=0)
            
            # Compute distances from mean
            distances = np.linalg.norm(current_X - current_mean, axis=1)
            
            # Determine threshold for trimming
            threshold = np.percentile(distances, (1 - trim_fraction) * 100)
            
            # Filter out outliers
            mask = distances <= threshold
            new_X = current_X[mask]
            
            # Check convergence
            if len(new_X) == len(current_X):
                # No more outliers to trim
                break
            
            current_X = new_X
            
            # Safety check: ensure we don't trim too much
            if len(current_X) < max(10, 0.1 * len(X)):
                break
        
        return np.mean(current_X, axis=0)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_observation_scores(
    X: np.ndarray,
    p_ref: Optional[np.ndarray] = None,
    weight_method: str = 'variance_ratio'
) -> Dict:
    """
    Compute unified observation scores using streaming computation.
    
    This is the main entry point for the framework.
    
    Args:
        X: Input data array of shape (N, m)
        p_ref: Reference signal of shape (m,). If None, computed as centroid
        weight_method: One of 'variance_ratio', 'iqr', 'cv', 'mutual_information', 'combined'
        
    Returns:
        Dictionary with:
            - log_action_scores: array of shape (N,) - unified log-action scores
            - component_scores: dict of metric name → array of shape (N,)
            - weights: dict of metric name → float (the α weights)
            - reference: the reference signal used
    """
    X = ensure_float64(X)
    validate_data(X, "X")
    
    N, m = X.shape
    
    # Compute reference if not provided
    if p_ref is None:
        p_ref = compute_reference(X, method='trimmed_mean')
    else:
        p_ref = ensure_float64(p_ref)
        validate_data(p_ref, "p_ref")
    
    # Convert reference to probability for KL and Hellinger
    p_ref_prob = normalize_to_probability(p_ref)
    
    # Initialize score arrays
    kl_scores = np.zeros(N, dtype=np.float64)
    lyapunov_scores = np.zeros(N, dtype=np.float64)
    spectral_scores = np.zeros(N, dtype=np.float64)
    hellinger_scores = np.zeros(N, dtype=np.float64)
    
    # STREAMING COMPUTATION: Process one observation at a time
    for i in range(N):
        observation = X[i, :]
        
        # Convert to probability for KL and Hellinger
        p_obs = normalize_to_probability(observation)
        
        # Compute all 4 metrics
        kl_scores[i] = kl_divergence(p_obs, p_ref_prob)
        lyapunov_scores[i] = lyapunov_exponent(observation, p_ref)
        spectral_scores[i] = spectral_divergence(observation, p_ref)
        hellinger_scores[i] = hellinger_distance(p_obs, p_ref_prob)
        
        # Note: FFT and probability vectors are garbage collected after each iteration
        # This ensures memory-safe operation even for large N
    
    # Store component scores
    component_scores = {
        'kl_divergence': kl_scores,
        'lyapunov_exponent': lyapunov_scores,
        'spectral_divergence': spectral_scores,
        'hellinger_distance': hellinger_scores,
    }
    
    # Compute centroid deviations for weight methods that need it
    centroid = np.mean(X, axis=0)
    centroid_deviations = np.linalg.norm(X - centroid, axis=1)
    
    # Compute weights based on selected method
    if weight_method == 'variance_ratio':
        weights = variance_ratio_weights(component_scores)
    elif weight_method == 'iqr':
        weights = iqr_weights(component_scores)
    elif weight_method == 'cv':
        weights = coefficient_of_variation_weights(component_scores)
    elif weight_method == 'mutual_information':
        weights = mutual_information_weights(component_scores, centroid_deviations)
    elif weight_method == 'combined':
        weights = combined_weights(component_scores, centroid_deviations)
    else:
        raise ValueError(f"Unknown weight_method: {weight_method}")
    
    # Compute unified log-action scores
    # log_A_i = α₁·D_kl + α₂·λ_i + α₃·D_freq + α₄·d_H
    log_action_scores = (
        weights['kl_divergence'] * kl_scores +
        weights['lyapunov_exponent'] * lyapunov_scores +
        weights['spectral_divergence'] * spectral_scores +
        weights['hellinger_distance'] * hellinger_scores
    )
    
    return {
        'log_action_scores': log_action_scores,
        'component_scores': component_scores,
        'weights': weights,
        'reference': p_ref,
    }

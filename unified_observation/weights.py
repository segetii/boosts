"""
Data-driven weight selection methods for the Unified Observation Framework.

Implements multiple methods to derive α weights from the data itself.
"""

import numpy as np
from typing import Dict
from sklearn.metrics import mutual_info_score
from .utils import ensure_float64


def variance_ratio_weights(scores_dict: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Compute weights proportional to variance of each metric.
    
    Formula:
        α_k = Var(D_k) / Σ Var(D_k')
    
    Args:
        scores_dict: Dictionary mapping metric name to array of scores
        
    Returns:
        Dictionary mapping metric name to weight (sum to 1)
    """
    variances = {}
    for name, scores in scores_dict.items():
        scores = ensure_float64(scores)
        variances[name] = np.var(scores)
    
    total_var = sum(variances.values())
    
    if total_var == 0:
        # Uniform weights if all variances are zero
        n = len(scores_dict)
        return {name: 1.0 / n for name in scores_dict.keys()}
    
    weights = {name: var / total_var for name, var in variances.items()}
    
    return weights


def iqr_weights(scores_dict: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Compute weights proportional to interquartile range (robust to outliers).
    
    Formula:
        α_k = IQR(D_k) / Σ IQR(D_k')
    
    Args:
        scores_dict: Dictionary mapping metric name to array of scores
        
    Returns:
        Dictionary mapping metric name to weight (sum to 1)
    """
    iqrs = {}
    for name, scores in scores_dict.items():
        scores = ensure_float64(scores)
        q75 = np.percentile(scores, 75)
        q25 = np.percentile(scores, 25)
        iqrs[name] = q75 - q25
    
    total_iqr = sum(iqrs.values())
    
    if total_iqr == 0:
        # Uniform weights if all IQRs are zero
        n = len(scores_dict)
        return {name: 1.0 / n for name in scores_dict.keys()}
    
    weights = {name: iqr / total_iqr for name, iqr in iqrs.items()}
    
    return weights


def coefficient_of_variation_weights(scores_dict: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Compute weights proportional to coefficient of variation (CV = σ/μ).
    
    Scale-normalized measure of dispersion.
    
    Args:
        scores_dict: Dictionary mapping metric name to array of scores
        
    Returns:
        Dictionary mapping metric name to weight (sum to 1)
    """
    cvs = {}
    for name, scores in scores_dict.items():
        scores = ensure_float64(scores)
        mean = np.mean(scores)
        std = np.std(scores)
        
        if mean == 0:
            cvs[name] = 0.0
        else:
            cvs[name] = std / abs(mean)
    
    total_cv = sum(cvs.values())
    
    if total_cv == 0:
        # Uniform weights if all CVs are zero
        n = len(scores_dict)
        return {name: 1.0 / n for name in scores_dict.keys()}
    
    weights = {name: cv / total_cv for name, cv in cvs.items()}
    
    return weights


def mutual_information_weights(
    scores_dict: Dict[str, np.ndarray],
    centroid_deviations: np.ndarray
) -> Dict[str, float]:
    """
    Compute weights proportional to mutual information with centroid deviation.
    
    Uses histogram-based MI estimation.
    
    Args:
        scores_dict: Dictionary mapping metric name to array of scores
        centroid_deviations: Array of distances from centroid for each observation
        
    Returns:
        Dictionary mapping metric name to weight (sum to 1)
    """
    mis = {}
    
    # Discretize centroid deviations for MI computation
    n_bins = min(20, len(centroid_deviations) // 10)
    if n_bins < 2:
        n_bins = 2
    
    centroid_discrete = np.digitize(centroid_deviations, 
                                    bins=np.percentile(centroid_deviations, 
                                                      np.linspace(0, 100, n_bins)))
    
    for name, scores in scores_dict.items():
        scores = ensure_float64(scores)
        
        # Discretize scores
        scores_discrete = np.digitize(scores, 
                                     bins=np.percentile(scores, 
                                                       np.linspace(0, 100, n_bins)))
        
        # Compute mutual information
        mi = mutual_info_score(scores_discrete, centroid_discrete)
        mis[name] = mi
    
    total_mi = sum(mis.values())
    
    if total_mi == 0:
        # Uniform weights if all MIs are zero
        n = len(scores_dict)
        return {name: 1.0 / n for name in scores_dict.keys()}
    
    weights = {name: mi / total_mi for name, mi in mis.items()}
    
    return weights


def fisher_weights(
    scores_dict: Dict[str, np.ndarray],
    labels_inner_outer: np.ndarray
) -> Dict[str, float]:
    """
    Compute Fisher discriminant ratio weights.
    
    Requires binary labels (inner/outer based on centroid distance).
    
    Args:
        scores_dict: Dictionary mapping metric name to array of scores
        labels_inner_outer: Binary labels (0 for inner, 1 for outer)
        
    Returns:
        Dictionary mapping metric name to weight (sum to 1)
    """
    fisher_ratios = {}
    
    for name, scores in scores_dict.items():
        scores = ensure_float64(scores)
        
        # Split by labels
        inner_scores = scores[labels_inner_outer == 0]
        outer_scores = scores[labels_inner_outer == 1]
        
        if len(inner_scores) == 0 or len(outer_scores) == 0:
            fisher_ratios[name] = 0.0
            continue
        
        # Compute means and variances
        mean_inner = np.mean(inner_scores)
        mean_outer = np.mean(outer_scores)
        var_inner = np.var(inner_scores)
        var_outer = np.var(outer_scores)
        
        # Fisher discriminant ratio: (μ1 - μ2)² / (σ1² + σ2²)
        numerator = (mean_inner - mean_outer)**2
        denominator = var_inner + var_outer
        
        if denominator == 0:
            fisher_ratios[name] = 0.0
        else:
            fisher_ratios[name] = numerator / denominator
    
    total_fisher = sum(fisher_ratios.values())
    
    if total_fisher == 0:
        # Uniform weights if all ratios are zero
        n = len(scores_dict)
        return {name: 1.0 / n for name in scores_dict.keys()}
    
    weights = {name: ratio / total_fisher for name, ratio in fisher_ratios.items()}
    
    return weights


def combined_weights(
    scores_dict: Dict[str, np.ndarray],
    centroid_deviations: np.ndarray
) -> Dict[str, float]:
    """
    Recommended default: combine variance and mutual information.
    
    Formula:
        α_k ∝ Var(D_k) · MI(D_k, δ)
    
    Args:
        scores_dict: Dictionary mapping metric name to array of scores
        centroid_deviations: Array of distances from centroid for each observation
        
    Returns:
        Dictionary mapping metric name to weight (sum to 1)
    """
    # Get variance weights
    var_weights = variance_ratio_weights(scores_dict)
    
    # Get MI weights
    mi_weights = mutual_information_weights(scores_dict, centroid_deviations)
    
    # Combine by multiplication
    combined = {}
    for name in scores_dict.keys():
        combined[name] = var_weights[name] * mi_weights[name]
    
    total = sum(combined.values())
    
    if total == 0:
        # Uniform weights if all combined values are zero
        n = len(scores_dict)
        return {name: 1.0 / n for name in scores_dict.keys()}
    
    weights = {name: val / total for name, val in combined.items()}
    
    return weights

"""
Demo: Synthetic Data with Known Normal and Mimic Distributions

Generate synthetic multivariate data with known normal and mimic distributions.
Tests that the framework correctly separates them.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unified_observation.scoring import compute_observation_scores
from unified_observation.classification import classify


def generate_normal_data(n_samples, n_features, mean=0.0, std=1.0):
    """
    Generate normal (non-anomalous) data from a Gaussian distribution.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        mean: Mean of Gaussian
        std: Standard deviation
        
    Returns:
        Array of shape (n_samples, n_features)
    """
    return np.random.randn(n_samples, n_features) * std + mean


def generate_mimic_data(n_samples, n_features, mimic_type='shifted'):
    """
    Generate mimic (anomalous) data that subtly differs from normal.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        mimic_type: Type of mimic ('shifted', 'scaled', 'mixed_freq')
        
    Returns:
        Array of shape (n_samples, n_features)
    """
    if mimic_type == 'shifted':
        # Shifted mean
        return np.random.randn(n_samples, n_features) * 1.0 + 2.0
    
    elif mimic_type == 'scaled':
        # Different variance
        return np.random.randn(n_samples, n_features) * 2.5
    
    elif mimic_type == 'mixed_freq':
        # Add high-frequency oscillations
        base = np.random.randn(n_samples, n_features)
        t = np.linspace(0, 10, n_features)
        oscillation = np.sin(10 * t) * 0.5
        return base + oscillation
    
    elif mimic_type == 'heavy_tail':
        # Heavy-tailed distribution (Laplace)
        return np.random.laplace(0, 1.0, size=(n_samples, n_features))
    
    else:
        raise ValueError(f"Unknown mimic_type: {mimic_type}")


def main():
    print("=" * 80)
    print("Unified Observation Framework - Synthetic Data Demo")
    print("=" * 80)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Parameters
    n_normal = 800
    n_mimic = 200
    n_features = 100
    
    # Generate data
    print("\n1. Generating synthetic data...")
    print(f"   Normal samples: {n_normal}")
    print(f"   Mimic samples:  {n_mimic}")
    print(f"   Features:       {n_features}")
    
    normal_data = generate_normal_data(n_normal, n_features)
    mimic_data = generate_mimic_data(n_mimic, n_features, mimic_type='shifted')
    
    # Combine and shuffle
    X = np.vstack([normal_data, mimic_data])
    true_labels = np.hstack([np.zeros(n_normal), np.ones(n_mimic)]).astype(bool)
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    true_labels = true_labels[shuffle_idx]
    
    print(f"   Total samples:  {len(X)}")
    
    # Test different weight methods
    weight_methods = ['variance_ratio', 'iqr', 'cv', 'combined']
    
    print("\n2. Testing different weight methods...")
    
    for weight_method in weight_methods:
        print(f"\n   --- Weight Method: {weight_method} ---")
        
        # Compute scores
        results = compute_observation_scores(X, weight_method=weight_method)
        
        log_scores = results['log_action_scores']
        weights = results['weights']
        component_scores = results['component_scores']
        
        print(f"   Weights:")
        for metric, weight in weights.items():
            print(f"      {metric:25s}: {weight:.4f}")
        
        # Classify
        detected_labels, threshold = classify(log_scores, method='percentile', threshold_param=80)
        
        # Evaluate
        true_positives = np.sum(detected_labels & true_labels)
        false_positives = np.sum(detected_labels & ~true_labels)
        false_negatives = np.sum(~detected_labels & true_labels)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"   Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    
    # Detailed analysis with best method
    print("\n3. Detailed component-wise contribution analysis (combined method)...")
    
    results = compute_observation_scores(X, weight_method='combined')
    log_scores = results['log_action_scores']
    weights = results['weights']
    component_scores = results['component_scores']
    
    detected_labels, threshold = classify(log_scores, method='percentile', threshold_param=80)
    
    print("\n   Component statistics for detected anomalies:")
    for metric, scores in component_scores.items():
        anomaly_scores = scores[detected_labels]
        normal_scores = scores[~detected_labels]
        
        print(f"\n   {metric}:")
        print(f"      Anomaly - mean: {np.mean(anomaly_scores):.4f}, std: {np.std(anomaly_scores):.4f}")
        print(f"      Normal  - mean: {np.mean(normal_scores):.4f}, std: {np.std(normal_scores):.4f}")
        print(f"      Separation: {(np.mean(anomaly_scores) - np.mean(normal_scores)) / np.std(normal_scores):.2f} σ")
    
    # Find which component contributed most
    print("\n4. Top contributing components for detected anomalies...")
    
    # Compute weighted contribution of each component
    contributions = {}
    for metric, scores in component_scores.items():
        weighted_scores = scores * weights[metric]
        contributions[metric] = np.mean(weighted_scores[detected_labels])
    
    sorted_contributions = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
    
    print("\n   Ranked by contribution:")
    for i, (metric, contrib) in enumerate(sorted_contributions, 1):
        print(f"      {i}. {metric:25s}: {contrib:.4f}")
    
    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

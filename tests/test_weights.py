"""
Tests for weight selection methods.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unified_observation.weights import (
    variance_ratio_weights,
    iqr_weights,
    coefficient_of_variation_weights,
    mutual_information_weights,
    fisher_weights,
    combined_weights,
)


def test_variance_ratio_weights():
    """Test variance ratio weights."""
    print("Testing variance_ratio_weights...")
    
    # Create synthetic scores with different variances
    scores_dict = {
        'metric1': np.random.randn(100) * 1.0,  # Low variance
        'metric2': np.random.randn(100) * 5.0,  # High variance
        'metric3': np.random.randn(100) * 2.0,  # Medium variance
    }
    
    weights = variance_ratio_weights(scores_dict)
    
    # Check weights sum to 1
    assert np.isclose(sum(weights.values()), 1.0), "Weights should sum to 1"
    
    # Check all weights are non-negative
    assert all(w >= 0 for w in weights.values()), "All weights should be non-negative"
    
    # Check high variance metric gets higher weight
    assert weights['metric2'] > weights['metric1'], "Higher variance should get higher weight"
    
    print("✓ variance_ratio_weights passed")


def test_iqr_weights():
    """Test IQR weights."""
    print("Testing iqr_weights...")
    
    # Create synthetic scores with different IQRs
    scores_dict = {
        'metric1': np.concatenate([np.ones(50), np.ones(50) * 2]),  # Low IQR
        'metric2': np.random.randn(100) * 5.0,  # High IQR
    }
    
    weights = iqr_weights(scores_dict)
    
    # Check weights sum to 1
    assert np.isclose(sum(weights.values()), 1.0), "Weights should sum to 1"
    
    # Check all weights are non-negative
    assert all(w >= 0 for w in weights.values()), "All weights should be non-negative"
    
    print("✓ iqr_weights passed")


def test_coefficient_of_variation_weights():
    """Test CV weights."""
    print("Testing coefficient_of_variation_weights...")
    
    # Create synthetic scores
    scores_dict = {
        'metric1': np.random.randn(100) + 10,  # High mean, moderate std
        'metric2': np.random.randn(100) * 5.0 + 1,  # Lower mean, high std
    }
    
    weights = coefficient_of_variation_weights(scores_dict)
    
    # Check weights sum to 1
    assert np.isclose(sum(weights.values()), 1.0), "Weights should sum to 1"
    
    # Check all weights are non-negative
    assert all(w >= 0 for w in weights.values()), "All weights should be non-negative"
    
    print("✓ coefficient_of_variation_weights passed")


def test_mutual_information_weights():
    """Test mutual information weights."""
    print("Testing mutual_information_weights...")
    
    # Create synthetic scores
    centroid_deviations = np.concatenate([np.ones(50) * 1.0, np.ones(50) * 5.0])
    
    scores_dict = {
        'metric1': centroid_deviations + np.random.randn(100) * 0.1,  # Correlated
        'metric2': np.random.randn(100),  # Uncorrelated
    }
    
    weights = mutual_information_weights(scores_dict, centroid_deviations)
    
    # Check weights sum to 1
    assert np.isclose(sum(weights.values()), 1.0), "Weights should sum to 1"
    
    # Check all weights are non-negative
    assert all(w >= 0 for w in weights.values()), "All weights should be non-negative"
    
    # Correlated metric should get higher weight
    assert weights['metric1'] >= weights['metric2'], "Correlated metric should get higher weight"
    
    print("✓ mutual_information_weights passed")


def test_fisher_weights():
    """Test Fisher weights."""
    print("Testing fisher_weights...")
    
    # Create synthetic scores with clear separation
    labels = np.array([0] * 50 + [1] * 50)
    
    scores_dict = {
        'metric1': np.concatenate([np.ones(50) * 1.0, np.ones(50) * 5.0]),  # Clear separation
        'metric2': np.random.randn(100),  # No separation
    }
    
    weights = fisher_weights(scores_dict, labels)
    
    # Check weights sum to 1
    assert np.isclose(sum(weights.values()), 1.0), "Weights should sum to 1"
    
    # Check all weights are non-negative
    assert all(w >= 0 for w in weights.values()), "All weights should be non-negative"
    
    # Well-separated metric should get higher weight
    assert weights['metric1'] > weights['metric2'], "Well-separated metric should get higher weight"
    
    print("✓ fisher_weights passed")


def test_combined_weights():
    """Test combined weights."""
    print("Testing combined_weights...")
    
    # Create synthetic scores
    centroid_deviations = np.random.randn(100) * 2.0
    
    scores_dict = {
        'metric1': np.random.randn(100) * 1.0,
        'metric2': np.random.randn(100) * 5.0,
        'metric3': np.random.randn(100) * 2.0,
    }
    
    weights = combined_weights(scores_dict, centroid_deviations)
    
    # Check weights sum to 1
    assert np.isclose(sum(weights.values()), 1.0), "Weights should sum to 1"
    
    # Check all weights are non-negative
    assert all(w >= 0 for w in weights.values()), "All weights should be non-negative"
    
    print("✓ combined_weights passed")


def run_all_tests():
    """Run all weight tests."""
    print("\n" + "=" * 80)
    print("Running Weight Tests")
    print("=" * 80 + "\n")
    
    test_variance_ratio_weights()
    test_iqr_weights()
    test_coefficient_of_variation_weights()
    test_mutual_information_weights()
    test_fisher_weights()
    test_combined_weights()
    
    print("\n" + "=" * 80)
    print("All weight tests passed!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_all_tests()

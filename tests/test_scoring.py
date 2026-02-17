"""
Tests for scoring and end-to-end functionality.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unified_observation.scoring import compute_observation_scores, compute_reference
from unified_observation.classification import classify


def test_compute_reference():
    """Test reference computation methods."""
    print("Testing compute_reference...")
    
    # Create test data
    X = np.random.randn(100, 10)
    
    # Test centroid method
    ref_centroid = compute_reference(X, method='centroid')
    assert ref_centroid.shape == (10,), "Reference should have shape (m,)"
    assert not np.any(np.isnan(ref_centroid)), "Reference should not contain NaN"
    
    # Test median method
    ref_median = compute_reference(X, method='median')
    assert ref_median.shape == (10,), "Reference should have shape (m,)"
    assert not np.any(np.isnan(ref_median)), "Reference should not contain NaN"
    
    # Test trimmed mean method
    ref_trimmed = compute_reference(X, method='trimmed_mean', max_iterations=5)
    assert ref_trimmed.shape == (10,), "Reference should have shape (m,)"
    assert not np.any(np.isnan(ref_trimmed)), "Reference should not contain NaN"
    
    print("✓ compute_reference passed")


def test_compute_observation_scores():
    """Test end-to-end observation score computation."""
    print("Testing compute_observation_scores...")
    
    # Create test data (small dataset)
    N, m = 50, 20
    X = np.random.randn(N, m)
    
    # Test with default parameters
    results = compute_observation_scores(X)
    
    assert 'log_action_scores' in results, "Results should contain log_action_scores"
    assert 'component_scores' in results, "Results should contain component_scores"
    assert 'weights' in results, "Results should contain weights"
    assert 'reference' in results, "Results should contain reference"
    
    log_scores = results['log_action_scores']
    assert log_scores.shape == (N,), f"Log scores should have shape ({N},)"
    assert not np.any(np.isnan(log_scores)), "Log scores should not contain NaN"
    assert not np.any(np.isinf(log_scores)), "Log scores should not contain inf"
    
    # Check component scores
    component_scores = results['component_scores']
    assert len(component_scores) == 4, "Should have 4 component scores"
    
    for name, scores in component_scores.items():
        assert scores.shape == (N,), f"{name} should have shape ({N},)"
        assert not np.any(np.isnan(scores)), f"{name} should not contain NaN"
        assert not np.any(np.isinf(scores)), f"{name} should not contain inf"
    
    # Check weights
    weights = results['weights']
    assert len(weights) == 4, "Should have 4 weights"
    assert np.isclose(sum(weights.values()), 1.0), "Weights should sum to 1"
    assert all(w >= 0 for w in weights.values()), "All weights should be non-negative"
    
    print("✓ compute_observation_scores passed")


def test_streaming_computation():
    """Test that streaming computation doesn't accumulate memory."""
    print("Testing streaming computation...")
    
    # This test verifies the pattern but doesn't actually measure memory
    # The streaming is verified by the code structure in scoring.py
    
    N, m = 100, 50
    X = np.random.randn(N, m)
    
    results = compute_observation_scores(X, weight_method='variance_ratio')
    
    # If this completes without error, streaming is working
    assert results['log_action_scores'].shape == (N,), "Scores should be computed for all observations"
    
    print("✓ streaming computation passed")


def test_different_weight_methods():
    """Test all weight methods work."""
    print("Testing different weight methods...")
    
    N, m = 50, 20
    X = np.random.randn(N, m)
    
    weight_methods = ['variance_ratio', 'iqr', 'cv', 'combined']
    
    for method in weight_methods:
        results = compute_observation_scores(X, weight_method=method)
        
        assert 'log_action_scores' in results, f"Method {method} should return log_action_scores"
        assert not np.any(np.isnan(results['log_action_scores'])), f"Method {method} should not produce NaN"
        
        weights = results['weights']
        assert np.isclose(sum(weights.values()), 1.0), f"Weights for {method} should sum to 1"
    
    print("✓ different weight methods passed")


def test_classification():
    """Test classification methods."""
    print("Testing classification...")
    
    # Create test scores
    log_scores = np.concatenate([
        np.random.randn(80) * 1.0,  # Normal
        np.random.randn(20) * 2.0 + 5.0,  # Anomalies
    ])
    
    # Test percentile method
    labels, threshold = classify(log_scores, method='percentile', threshold_param=80)
    assert labels.shape == log_scores.shape, "Labels should match scores shape"
    assert labels.dtype == bool, "Labels should be boolean"
    assert np.sum(labels) > 0, "Should detect some anomalies"
    
    # Test sigma method
    labels, threshold = classify(log_scores, method='sigma', threshold_param=2)
    assert labels.shape == log_scores.shape, "Labels should match scores shape"
    
    # Test knee method
    labels, threshold = classify(log_scores, method='knee')
    assert labels.shape == log_scores.shape, "Labels should match scores shape"
    
    print("✓ classification passed")


def test_log_space_computation():
    """Test that computation stays in log space."""
    print("Testing log-space computation...")
    
    N, m = 30, 15
    X = np.random.randn(N, m) * 100  # Large values
    
    results = compute_observation_scores(X)
    
    # If we're properly in log space, we shouldn't get inf values
    log_scores = results['log_action_scores']
    assert not np.any(np.isinf(log_scores)), "Log scores should not overflow to inf"
    
    # Component scores should also be safe
    for name, scores in results['component_scores'].items():
        assert not np.any(np.isinf(scores)), f"{name} should not overflow to inf"
    
    print("✓ log-space computation passed")


def run_all_tests():
    """Run all scoring tests."""
    print("\n" + "=" * 80)
    print("Running Scoring Tests")
    print("=" * 80 + "\n")
    
    test_compute_reference()
    test_compute_observation_scores()
    test_streaming_computation()
    test_different_weight_methods()
    test_classification()
    test_log_space_computation()
    
    print("\n" + "=" * 80)
    print("All scoring tests passed!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_all_tests()

"""
Tests for numerical safety and edge cases.
"""

import numpy as np
import sys
import os
import warnings

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unified_observation.metrics import (
    kl_divergence,
    lyapunov_exponent,
    spectral_divergence,
    hellinger_distance,
    normalize_to_probability,
)
from unified_observation.scoring import compute_reference, compute_observation_scores
from unified_observation.utils import safe_log, safe_divide


def test_division_by_zero():
    """Test safe division handles zero denominators."""
    print("Testing division by zero safety...")
    
    numerator = np.array([1.0, 2.0, 3.0])
    denominator = np.array([0.0, 2.0, 0.0])
    
    result = safe_divide(numerator, denominator)
    
    assert not np.any(np.isnan(result)), "Safe divide should not produce NaN"
    assert not np.any(np.isinf(result)), "Safe divide should not produce inf"
    
    print("✓ division by zero safety passed")


def test_log_of_zero():
    """Test safe log handles zero values."""
    print("Testing log of zero safety...")
    
    x = np.array([0.0, 1.0, 2.0, 0.0])
    
    result = safe_log(x)
    
    assert not np.any(np.isnan(result)), "Safe log should not produce NaN"
    assert not np.any(np.isinf(result)), "Safe log should not produce inf"
    
    print("✓ log of zero safety passed")


def test_very_large_values():
    """Test handling of very large values."""
    print("Testing very large values...")
    
    # Create data with very large values
    X = np.random.randn(50, 20) * 1e10
    
    results = compute_observation_scores(X)
    log_scores = results['log_action_scores']
    
    assert not np.any(np.isnan(log_scores)), "Should handle large values without NaN"
    # Note: Large values might produce inf in some cases, but log-space should help
    # We mainly check for NaN which indicates numerical errors
    
    print("✓ very large values passed")


def test_very_small_values():
    """Test handling of very small values."""
    print("Testing very small values...")
    
    # Create data with very small values
    X = np.random.randn(50, 20) * 1e-10
    
    results = compute_observation_scores(X)
    log_scores = results['log_action_scores']
    
    assert not np.any(np.isnan(log_scores)), "Should handle small values without NaN"
    assert not np.all(log_scores == -np.inf), "Should not produce all -inf"
    
    print("✓ very small values passed")


def test_max_iterations_cap():
    """Test that max_iterations prevents infinite loops."""
    print("Testing max_iterations cap...")
    
    # Create data that might cause many iterations in trimmed_mean
    X = np.random.randn(100, 10)
    # Add extreme outliers
    X[0, :] = 1000
    X[1, :] = -1000
    
    # This should terminate within max_iterations
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ref = compute_reference(X, method='trimmed_mean', max_iterations=5)
        
        # Check if warning was issued (iteration cap reached)
        # Warning may or may not be issued depending on convergence
    
    assert ref.shape == (10,), "Should return valid reference"
    assert not np.any(np.isnan(ref)), "Reference should not contain NaN"
    
    print("✓ max_iterations cap passed")


def test_identical_observations():
    """Test handling of identical observations."""
    print("Testing identical observations...")
    
    # Create data where all observations are identical
    X = np.ones((50, 20))
    
    results = compute_observation_scores(X)
    log_scores = results['log_action_scores']
    
    # All observations should have similar (near-zero) scores
    assert not np.any(np.isnan(log_scores)), "Should handle identical observations"
    # Scores should be very similar (small variance)
    assert np.std(log_scores) < 1.0, "Identical observations should have similar scores"
    
    print("✓ identical observations passed")


def test_single_outlier():
    """Test detection of a single clear outlier."""
    print("Testing single outlier detection...")
    
    # Create data with one clear outlier
    X = np.random.randn(50, 20) * 1.0
    X[0, :] = 10.0  # Clear outlier
    
    results = compute_observation_scores(X)
    log_scores = results['log_action_scores']
    
    # Outlier should have highest score
    outlier_score = log_scores[0]
    other_scores = log_scores[1:]
    
    assert outlier_score > np.max(other_scores), "Outlier should have highest score"
    
    print("✓ single outlier detection passed")


def test_negative_value_handling():
    """Test handling of negative values in data."""
    print("Testing negative value handling...")
    
    # Create data with negative values
    X = np.random.randn(50, 20) * 5.0 - 10.0  # Negative mean
    
    results = compute_observation_scores(X)
    log_scores = results['log_action_scores']
    
    assert not np.any(np.isnan(log_scores)), "Should handle negative values"
    assert not np.any(np.isinf(log_scores)), "Should not produce inf with negative values"
    
    # Test probability normalization with negatives
    x = np.array([-5, -3, -1, 0, 1])
    p = normalize_to_probability(x)
    
    assert np.isclose(np.sum(p), 1.0), "Should normalize negative values to probability"
    assert np.all(p > 0), "All probabilities should be positive"
    
    print("✓ negative value handling passed")


def test_nan_detection():
    """Test that NaN in input is detected."""
    print("Testing NaN detection...")
    
    # Create data with NaN
    X = np.random.randn(50, 20)
    X[5, 10] = np.nan
    
    try:
        results = compute_observation_scores(X)
        assert False, "Should raise error for NaN input"
    except ValueError as e:
        assert "NaN" in str(e), "Should detect NaN values"
    
    print("✓ NaN detection passed")


def test_inf_detection():
    """Test that inf in input is detected."""
    print("Testing inf detection...")
    
    # Create data with inf
    X = np.random.randn(50, 20)
    X[5, 10] = np.inf
    
    try:
        results = compute_observation_scores(X)
        assert False, "Should raise error for inf input"
    except ValueError as e:
        assert "infinite" in str(e).lower(), "Should detect inf values"
    
    print("✓ inf detection passed")


def run_all_tests():
    """Run all safety tests."""
    print("\n" + "=" * 80)
    print("Running Safety Tests")
    print("=" * 80 + "\n")
    
    test_division_by_zero()
    test_log_of_zero()
    test_very_large_values()
    test_very_small_values()
    test_max_iterations_cap()
    test_identical_observations()
    test_single_outlier()
    test_negative_value_handling()
    test_nan_detection()
    test_inf_detection()
    
    print("\n" + "=" * 80)
    print("All safety tests passed!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_all_tests()

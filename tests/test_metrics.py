"""
Tests for metric functions.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unified_observation.metrics import (
    normalize_to_probability,
    kl_divergence,
    lyapunov_exponent,
    spectral_divergence,
    hellinger_distance,
)


def test_normalize_to_probability():
    """Test probability normalization."""
    print("Testing normalize_to_probability...")
    
    # Test basic normalization
    x = np.array([1.0, 2.0, 3.0, 4.0])
    p = normalize_to_probability(x)
    
    assert np.isclose(np.sum(p), 1.0), "Probability should sum to 1"
    assert np.all(p > 0), "All probabilities should be positive"
    
    # Test negative values
    x_neg = np.array([-2.0, -1.0, 0.0, 1.0])
    p_neg = normalize_to_probability(x_neg)
    
    assert np.isclose(np.sum(p_neg), 1.0), "Probability should sum to 1 even with negatives"
    assert np.all(p_neg > 0), "All probabilities should be positive after shifting"
    
    # Test all zeros
    x_zero = np.zeros(10)
    p_zero = normalize_to_probability(x_zero)
    
    assert np.isclose(np.sum(p_zero), 1.0), "Uniform distribution for all zeros"
    assert np.allclose(p_zero, 1.0/10), "Should be uniform"
    
    print("✓ normalize_to_probability passed")


def test_kl_divergence():
    """Test KL divergence."""
    print("Testing kl_divergence...")
    
    # Test identical distributions
    p = np.array([0.25, 0.25, 0.25, 0.25])
    kl = kl_divergence(p, p)
    
    assert kl >= 0, "KL divergence should be non-negative"
    assert kl < 1e-6, "KL divergence of identical distributions should be near zero"
    
    # Test different distributions
    p1 = np.array([0.5, 0.3, 0.15, 0.05])
    p2 = np.array([0.25, 0.25, 0.25, 0.25])
    kl = kl_divergence(p1, p2)
    
    assert kl > 0, "KL divergence of different distributions should be positive"
    
    # Test epsilon smoothing prevents NaN
    p_with_zero = np.array([0.0, 0.5, 0.5])
    p_ref = np.array([0.33, 0.33, 0.34])
    kl = kl_divergence(p_with_zero, p_ref)
    
    assert not np.isnan(kl), "KL divergence should not produce NaN with zeros"
    assert not np.isinf(kl), "KL divergence should not produce inf with zeros"
    
    print("✓ kl_divergence passed")


def test_lyapunov_exponent():
    """Test Lyapunov exponent."""
    print("Testing lyapunov_exponent...")
    
    # Test identical signals
    f = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    lyap = lyapunov_exponent(f, f)
    
    assert not np.isnan(lyap), "Lyapunov should not be NaN for identical signals"
    assert not np.isinf(lyap), "Lyapunov should not be inf for identical signals"
    
    # Test diverging signals
    f1 = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    f2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    lyap = lyapunov_exponent(f1, f2)
    
    assert not np.isnan(lyap), "Lyapunov should not be NaN"
    assert not np.isinf(lyap), "Lyapunov should not be inf"
    
    # Test epsilon protection
    f_close = np.array([1.0, 1.0001, 1.0002, 1.0003])
    f_ref = np.array([1.0, 1.0, 1.0, 1.0])
    lyap = lyapunov_exponent(f_close, f_ref)
    
    assert not np.isnan(lyap), "Epsilon should prevent NaN"
    assert not np.isinf(lyap), "Epsilon should prevent inf"
    
    print("✓ lyapunov_exponent passed")


def test_spectral_divergence():
    """Test spectral divergence."""
    print("Testing spectral_divergence...")
    
    # Test identical signals
    f = np.sin(np.linspace(0, 4*np.pi, 100))
    spec_div = spectral_divergence(f, f)
    
    assert spec_div >= 0, "Spectral divergence should be non-negative"
    assert spec_div < 1e-10, "Spectral divergence of identical signals should be near zero"
    
    # Test different signals
    f1 = np.sin(np.linspace(0, 4*np.pi, 100))
    f2 = np.sin(np.linspace(0, 8*np.pi, 100))  # Different frequency
    spec_div = spectral_divergence(f1, f2)
    
    assert spec_div > 0, "Spectral divergence of different signals should be positive"
    
    print("✓ spectral_divergence passed")


def test_hellinger_distance():
    """Test Hellinger distance."""
    print("Testing hellinger_distance...")
    
    # Test identical distributions
    p = np.array([0.25, 0.25, 0.25, 0.25])
    hd = hellinger_distance(p, p)
    
    assert hd >= 0, "Hellinger distance should be non-negative"
    assert hd < 1e-10, "Hellinger distance of identical distributions should be near zero"
    
    # Test different distributions
    p1 = np.array([1.0, 0.0, 0.0, 0.0])
    p2 = np.array([0.0, 0.0, 0.0, 1.0])
    hd = hellinger_distance(p1, p2)
    
    assert hd > 0, "Hellinger distance of different distributions should be positive"
    assert hd <= 1.0, "Hellinger distance should be bounded by 1"
    
    print("✓ hellinger_distance passed")


def run_all_tests():
    """Run all metric tests."""
    print("\n" + "=" * 80)
    print("Running Metric Tests")
    print("=" * 80 + "\n")
    
    test_normalize_to_probability()
    test_kl_divergence()
    test_lyapunov_exponent()
    test_spectral_divergence()
    test_hellinger_distance()
    
    print("\n" + "=" * 80)
    print("All metric tests passed!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_all_tests()

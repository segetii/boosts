# Project Structure

```
boosts/
├── unified_observation/          # Core package
│   ├── __init__.py              # Package initialization
│   ├── metrics.py               # Core metric functions (KL, Lyapunov, Spectral, Hellinger)
│   ├── weights.py               # Data-driven weight selection (6 methods)
│   ├── scoring.py               # Unified action score computation (streaming)
│   ├── classification.py        # Anomaly classification (percentile, sigma, knee)
│   ├── clustering.py            # Memory-safe clustering (FAISS support)
│   └── utils.py                 # Safety utilities and helpers
│
├── examples/                     # Demonstration scripts
│   ├── demo_lorenz.py           # Lorenz attractor (chaotic system) demo
│   └── demo_synthetic.py        # Synthetic data validation demo
│
├── tests/                        # Comprehensive test suite
│   ├── test_metrics.py          # Metric function tests
│   ├── test_weights.py          # Weight computation tests
│   ├── test_scoring.py          # End-to-end integration tests
│   └── test_safety.py           # Numerical safety tests
│
├── requirements.txt              # Python dependencies
├── README.md                     # Main documentation
├── STRUCTURE.md                  # This file
└── .gitignore                    # Git ignore rules
```

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run tests**:
   ```bash
   python tests/test_metrics.py
   python tests/test_weights.py
   python tests/test_scoring.py
   python tests/test_safety.py
   ```

3. **Try the demos**:
   ```bash
   python examples/demo_lorenz.py
   python examples/demo_synthetic.py
   ```

4. **Use in your code**:
   ```python
   from unified_observation.scoring import compute_observation_scores
   from unified_observation.classification import classify
   
   # Your data: N observations, m features each
   import numpy as np
   X = np.random.randn(1000, 100)
   
   # Compute scores
   results = compute_observation_scores(X, weight_method='combined')
   
   # Classify anomalies
   labels, threshold = classify(results['log_action_scores'], method='percentile', threshold_param=95)
   ```

## Module Overview

### `unified_observation/metrics.py`
Core mathematical functions:
- `normalize_to_probability()` - Convert raw features to probability distribution
- `kl_divergence()` - KL divergence between distributions
- `lyapunov_exponent()` - Pseudo-Lyapunov exponent (dynamic divergence)
- `spectral_divergence()` - FFT-based frequency domain divergence
- `hellinger_distance()` - Geometric distance on probability manifold

### `unified_observation/weights.py`
Data-driven weight computation:
- `variance_ratio_weights()` - Weights proportional to variance
- `iqr_weights()` - Weights proportional to IQR (robust to outliers)
- `coefficient_of_variation_weights()` - Weights proportional to CV (σ/μ)
- `mutual_information_weights()` - Weights based on MI with centroid
- `fisher_weights()` - Fisher discriminant ratio weights
- `combined_weights()` - **Recommended**: Variance × MI

### `unified_observation/scoring.py`
Main entry point:
- `compute_reference()` - Compute robust reference signal (centroid, median, trimmed mean)
- `compute_observation_scores()` - **Main function**: compute unified scores with streaming

### `unified_observation/classification.py`
Anomaly detection:
- `classify()` - Threshold-based classification (percentile, sigma, knee methods)

### `unified_observation/clustering.py`
Memory-safe clustering:
- `cluster_observations()` - FAISS-based clustering for large N, sklearn for small N

### `unified_observation/utils.py`
Utilities:
- `safe_log()` - Log with epsilon protection
- `safe_divide()` - Division with epsilon protection
- `validate_data()` - Input validation (NaN, inf detection)
- `estimate_memory()` - Memory usage estimation
- `ensure_float64()` - Type conversion for numerical stability
- `handle_negative_values()` - Shift-and-normalize for negative inputs

## Design Principles

1. **Memory Safety**: Streaming computation, no full pairwise matrices
2. **Numerical Stability**: Epsilon smoothing, log-space arithmetic, float64
3. **Data-Driven**: No arbitrary hyperparameters, weights computed from data
4. **Modularity**: Clean separation between metrics, weights, scoring
5. **Testability**: Comprehensive test coverage for all components
6. **Robustness**: Handles edge cases (NaN, inf, zeros, negatives)

## Memory Budget

For N=100K observations, m=10K features:
- Raw data: ~8 GB
- Reference: ~80 KB
- Scores: ~4 MB
- FFT buffer: ~80 KB
- **Total streaming**: ~8.2 GB ✅ (safe for 50GB)
- Full pairwise (AVOID): ~80 GB 🔴

## All Safety Requirements Met

✅ Epsilon smoothing everywhere  
✅ Log-space computation  
✅ Streaming row-by-row  
✅ No full pairwise matrices for N > 10K  
✅ Capped iterations  
✅ No infinite loops  
✅ Negative value handling  
✅ Float64 throughout

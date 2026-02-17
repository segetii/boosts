# Unified Observation Framework

A mathematically grounded, non-ML anomaly and mimic detection system that combines multiple mathematical domains into a single unified action score.

## Overview

The Unified Observation Framework detects subtle mimics or anomalies by computing a unified action score that integrates four mathematical domains:

1. **Probability / KL Divergence** — statistical distance from a reference signal
2. **Chaos / Lyapunov Exponent** — dynamic divergence sensitivity
3. **Frequency / Spectral Divergence** — FFT-based pattern differences
4. **Geometry / Hellinger Distance** — manifold-aware distributional distance

These metrics are combined into a unified log-action score:

```
log_A_i = α₁·D_kl + α₂·λ_i + α₃·D_freq + α₄·d_H
```

Where the α weights are **derived from the data itself** (not arbitrary), using variance ratio, IQR, mutual information, or Fisher discriminant methods.

## Key Features

- ✅ **Memory-safe**: Designed for machines with 50 GB RAM / 50 GB GPU VRAM
- ✅ **Streaming computation**: Processes observations one at a time to minimize memory usage
- ✅ **Log-space arithmetic**: Prevents numerical overflow/underflow
- ✅ **Epsilon smoothing**: All divisions and logarithms are protected against numerical errors
- ✅ **Data-driven weights**: No arbitrary hyperparameters; weights computed from data statistics
- ✅ **Multiple weight methods**: Choose from variance ratio, IQR, CV, mutual information, Fisher, or combined
- ✅ **Flexible classification**: Percentile-based, sigma-based, or knee-point detection
- ✅ **Scalable clustering**: FAISS-based approximate clustering for large N

## Installation

```bash
# Clone the repository
git clone https://github.com/segetii/boosts.git
cd boosts

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
import numpy as np
from unified_observation.scoring import compute_observation_scores
from unified_observation.classification import classify

# Generate or load your data (N observations, m features each)
X = np.random.randn(1000, 100)

# Compute unified observation scores
results = compute_observation_scores(X, weight_method='combined')

# Extract results
log_scores = results['log_action_scores']  # Unified scores for each observation
weights = results['weights']  # The α weights used
component_scores = results['component_scores']  # Individual metric scores

# Classify anomalies
anomaly_labels, threshold = classify(log_scores, method='percentile', threshold_param=95)

# Print results
print(f"Detected {np.sum(anomaly_labels)} anomalies")
print(f"Weights: {weights}")
```

## API Reference

### Core Functions

#### `compute_observation_scores(X, p_ref=None, weight_method='variance_ratio')`

Compute unified observation scores using streaming computation.

**Parameters:**
- `X` (ndarray): Input data of shape (N, m)
- `p_ref` (ndarray, optional): Reference signal of shape (m,). If None, computed as robust centroid
- `weight_method` (str): One of 'variance_ratio', 'iqr', 'cv', 'mutual_information', 'combined'

**Returns:**
- Dictionary with:
  - `log_action_scores`: array of shape (N,) — unified log-action scores
  - `component_scores`: dict of metric name → array of shape (N,)
  - `weights`: dict of metric name → float
  - `reference`: the reference signal used

#### `classify(log_scores, method='percentile', threshold_param=95)`

Classify observations as normal or anomaly.

**Parameters:**
- `log_scores` (ndarray): Array of log-action scores
- `method` (str): One of 'percentile', 'sigma', 'knee'
- `threshold_param` (float): Threshold parameter (percentile value or sigma multiplier)

**Returns:**
- Tuple of (labels, threshold):
  - `labels`: Boolean array (True = anomaly)
  - `threshold`: The threshold value used

#### `cluster_observations(log_scores, method='faiss', n_clusters=None)`

Cluster observations using memory-safe methods.

**Parameters:**
- `log_scores` (ndarray): Array of log-action scores (N,) or features (N, d)
- `method` (str): One of 'faiss', 'dbscan', 'hierarchical'
- `n_clusters` (int, optional): Number of clusters

**Returns:**
- Tuple of (labels, info):
  - `labels`: Cluster labels
  - `info`: Dictionary with clustering metadata

### Metrics

All metrics are available in `unified_observation.metrics`:

- `kl_divergence(p, p_ref, eps=1e-12)` — KL divergence between probability distributions
- `lyapunov_exponent(f, f_ref, eps=1e-12)` — Pseudo-Lyapunov exponent
- `spectral_divergence(f, f_ref)` — FFT-based spectral divergence
- `hellinger_distance(p, p_ref)` — Hellinger distance
- `normalize_to_probability(x, eps=1e-12)` — Convert to probability distribution

### Weight Methods

All weight methods are available in `unified_observation.weights`:

- `variance_ratio_weights(scores_dict)` — Weights ∝ variance
- `iqr_weights(scores_dict)` — Weights ∝ IQR (robust to outliers)
- `coefficient_of_variation_weights(scores_dict)` — Weights ∝ CV (σ/μ)
- `mutual_information_weights(scores_dict, centroid_deviations)` — Weights ∝ MI with centroid
- `fisher_weights(scores_dict, labels_inner_outer)` — Fisher discriminant ratio
- `combined_weights(scores_dict, centroid_deviations)` — Recommended: Var × MI

## Examples

### Lorenz Attractor Demo

Demonstrates anomaly detection on a chaotic dynamical system:

```bash
python examples/demo_lorenz.py
```

This generates a Lorenz attractor trajectory, injects synthetic anomalies, and evaluates detection performance.

### Synthetic Data Demo

Tests the framework on synthetic multivariate data with known normal and mimic distributions:

```bash
python examples/demo_synthetic.py
```

This compares different weight methods and shows component-wise contribution analysis.

## Hardware Requirements

The framework is designed to be memory-safe for machines with:
- **50 GB RAM**
- **50 GB GPU VRAM** (though CPU-only operation is default)

### Memory Budget (Example: N=100K, m=10K)

| Component | Memory Usage |
|-----------|--------------|
| Raw data X | ~8 GB |
| Reference signal | ~80 KB |
| Score arrays | ~4 MB |
| FFT buffer (1 row) | ~80 KB |
| **Total (streaming)** | **~8.2 GB** ✅ |
| Full pairwise (AVOID) | ~80 GB 🔴 |

## Memory Safety Guarantees

1. **Epsilon smoothing everywhere**: Every `log()` and division has epsilon protection
2. **Log-space computation**: Stores and compares `log_A_i`, never exponentiates during computation
3. **Streaming row-by-row**: For per-observation metrics, processes one row at a time
4. **No full pairwise matrix for N > 10,000**: Uses FAISS or approximate methods
5. **Capped iterations**: Every loop has guaranteed termination via `max_iterations`
6. **Float64 throughout**: Maximizes numerical range before overflow

## Testing

Run all tests:

```bash
# Test individual metrics
python tests/test_metrics.py

# Test weight methods
python tests/test_weights.py

# Test scoring and end-to-end
python tests/test_scoring.py

# Test numerical safety
python tests/test_safety.py
```

All tests should pass with clear output indicating successful validation.

## Known Challenges and Status

| # | Challenge | Status |
|---|-----------|--------|
| 1 | Division by zero | ✅ Handled via epsilon in denominator |
| 2 | Log of zero | ✅ Handled via epsilon smoothing |
| 3 | Negative values in probability | ✅ Shift-and-normalize |
| 4 | Large N memory explosion | ✅ Streaming + FAISS clustering |
| 5 | FFT memory accumulation | ✅ Computed and discarded per row |
| 6 | Numerical overflow | ✅ Log-space + float64 |
| 7 | Numerical underflow | ✅ Epsilon protection |
| 8 | Infinite loops | ✅ max_iterations caps |
| 9 | NaN propagation | ✅ Input validation + epsilon |
| 10 | Arbitrary weights | ✅ Data-driven weight methods |
| 11 | Hyperparameter tuning | ✅ Minimal hyperparameters needed |
| 12 | Reference signal bias | ✅ Trimmed mean with outlier removal |
| 13 | Imbalanced data | ✅ Robust weight methods (IQR, Fisher) |
| 14 | Computational complexity | ✅ O(N·m) streaming, FAISS for clustering |

## Mathematical Background

### Unified Action Score

The framework computes a unified action score by combining four complementary views:

```
log_A_i = α₁·D_kl(p_i || p_ref) + α₂·λ_i(f_i, f_ref) + α₃·D_freq(f_i, f_ref) + α₄·d_H(p_i, p_ref)
```

Where:
- **D_kl**: KL divergence measuring probabilistic distance
- **λ_i**: Lyapunov exponent measuring dynamic sensitivity
- **D_freq**: Spectral divergence measuring frequency-domain differences
- **d_H**: Hellinger distance measuring geometric distance on probability manifold

### Weight Computation

Weights α are derived from data statistics, ensuring they reflect the relative discriminative power of each metric:

**Variance Ratio**: 
```
α_k = Var(D_k) / Σ_k' Var(D_k')
```

**Combined (Recommended)**:
```
α_k ∝ Var(D_k) · MI(D_k, δ)
```

Where MI is mutual information between metric scores and centroid deviations.

## Contributing

Contributions are welcome! Please ensure:
- All tests pass
- New features include tests
- Memory safety guarantees are maintained
- Code follows epsilon smoothing and log-space patterns

## License

[Add your license here]

## Citation

If you use this framework in your research, please cite:

```
[Add citation format here]
```
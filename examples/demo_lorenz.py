"""
Demo: Lorenz Attractor with Anomaly Detection

Generate Lorenz attractor data (known chaotic system) and inject synthetic anomalies.
Demonstrates the unified observation framework on a real chaotic dynamical system.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unified_observation.scoring import compute_observation_scores
from unified_observation.classification import classify


def lorenz(x, y, z, sigma=10, rho=28, beta=8/3, dt=0.01):
    """
    Compute the next step of the Lorenz attractor.
    
    Args:
        x, y, z: Current state
        sigma, rho, beta: Lorenz parameters
        dt: Time step
        
    Returns:
        Next state (x, y, z)
    """
    dx = sigma * (y - x) * dt
    dy = (x * (rho - z) - y) * dt
    dz = (x * y - beta * z) * dt
    
    return x + dx, y + dy, z + dz


def generate_lorenz_trajectory(n_steps, x0=1.0, y0=1.0, z0=1.0):
    """
    Generate a trajectory on the Lorenz attractor.
    
    Args:
        n_steps: Number of time steps
        x0, y0, z0: Initial conditions
        
    Returns:
        Array of shape (n_steps, 3) with trajectory
    """
    trajectory = np.zeros((n_steps, 3))
    x, y, z = x0, y0, z0
    
    for i in range(n_steps):
        trajectory[i] = [x, y, z]
        x, y, z = lorenz(x, y, z)
    
    return trajectory


def inject_anomalies(trajectory, n_anomalies=10, perturbation_scale=5.0):
    """
    Inject anomalies by perturbing random points in the trajectory.
    
    Args:
        trajectory: Original trajectory array
        n_anomalies: Number of anomalies to inject
        perturbation_scale: Scale of perturbation
        
    Returns:
        Modified trajectory and indices of anomalies
    """
    trajectory_copy = trajectory.copy()
    n_points = len(trajectory)
    
    # Randomly select points to perturb
    anomaly_indices = np.random.choice(n_points, size=n_anomalies, replace=False)
    
    for idx in anomaly_indices:
        # Add large random perturbation
        perturbation = np.random.randn(3) * perturbation_scale
        trajectory_copy[idx] += perturbation
    
    return trajectory_copy, anomaly_indices


def main():
    print("=" * 80)
    print("Unified Observation Framework - Lorenz Attractor Demo")
    print("=" * 80)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate Lorenz attractor trajectory
    print("\n1. Generating Lorenz attractor trajectory...")
    n_steps = 1000
    trajectory = generate_lorenz_trajectory(n_steps)
    print(f"   Generated {n_steps} points on Lorenz attractor")
    
    # Inject anomalies
    print("\n2. Injecting synthetic anomalies...")
    n_anomalies = 50
    trajectory_with_anomalies, true_anomaly_indices = inject_anomalies(
        trajectory, n_anomalies=n_anomalies, perturbation_scale=5.0
    )
    print(f"   Injected {n_anomalies} anomalies")
    
    # Compute observation scores
    print("\n3. Computing unified observation scores...")
    results = compute_observation_scores(
        trajectory_with_anomalies,
        weight_method='combined'
    )
    
    log_scores = results['log_action_scores']
    weights = results['weights']
    component_scores = results['component_scores']
    
    print(f"   Computed scores for {len(log_scores)} observations")
    print(f"\n   Weights (α):")
    for metric, weight in weights.items():
        print(f"      {metric:25s}: {weight:.4f}")
    
    # Classify anomalies
    print("\n4. Classifying anomalies...")
    detected_labels, threshold = classify(log_scores, method='percentile', threshold_param=95)
    n_detected = np.sum(detected_labels)
    print(f"   Detected {n_detected} anomalies using threshold {threshold:.4f}")
    
    # Evaluate detection performance
    print("\n5. Evaluating detection performance...")
    true_labels = np.zeros(n_steps, dtype=bool)
    true_labels[true_anomaly_indices] = True
    
    # Compute metrics
    true_positives = np.sum(detected_labels & true_labels)
    false_positives = np.sum(detected_labels & ~true_labels)
    false_negatives = np.sum(~detected_labels & true_labels)
    true_negatives = np.sum(~detected_labels & ~true_labels)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"   True Positives:  {true_positives}")
    print(f"   False Positives: {false_positives}")
    print(f"   False Negatives: {false_negatives}")
    print(f"   True Negatives:  {true_negatives}")
    print(f"\n   Precision: {precision:.3f}")
    print(f"   Recall:    {recall:.3f}")
    print(f"   F1 Score:  {f1:.3f}")
    
    # Component contribution analysis
    print("\n6. Component contribution analysis...")
    print("   Average component scores for anomalies vs normal:")
    
    for metric in component_scores.keys():
        scores = component_scores[metric]
        avg_anomaly = np.mean(scores[detected_labels])
        avg_normal = np.mean(scores[~detected_labels])
        ratio = avg_anomaly / avg_normal if avg_normal > 0 else float('inf')
        
        print(f"      {metric:25s}: anomaly={avg_anomaly:.4f}, normal={avg_normal:.4f}, ratio={ratio:.2f}x")
    
    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

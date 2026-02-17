"""
Clustering methods with memory safety for large datasets.

Uses approximate methods (FAISS) for large N to avoid full pairwise matrices.
"""

import numpy as np
from typing import Optional, Tuple
from .utils import ensure_float64


def cluster_observations(
    log_scores: np.ndarray,
    method: str = 'faiss',
    n_clusters: Optional[int] = None
) -> Tuple[np.ndarray, dict]:
    """
    Cluster observations using memory-safe methods.
    
    Args:
        log_scores: Array of log-action scores (N,) or features (N, d)
        method: One of 'faiss', 'dbscan', 'hierarchical'
        n_clusters: Number of clusters (for methods that need it)
        
    Returns:
        Tuple of (labels, info):
            - labels: Cluster labels for each observation
            - info: Dictionary with clustering metadata
    """
    log_scores = ensure_float64(log_scores)
    
    # Reshape to 2D if needed
    if log_scores.ndim == 1:
        log_scores = log_scores.reshape(-1, 1)
    
    N, d = log_scores.shape
    
    if method == 'faiss' and N > 10000:
        # Use FAISS for large datasets
        try:
            import faiss
            
            # Determine number of clusters if not provided
            if n_clusters is None:
                n_clusters = max(2, int(np.sqrt(N / 2)))
            
            # Normalize data for better clustering
            data_normalized = log_scores.astype(np.float32)
            data_normalized = data_normalized - np.mean(data_normalized, axis=0)
            std = np.std(data_normalized, axis=0)
            std[std == 0] = 1.0
            data_normalized = data_normalized / std
            
            # Use FAISS k-means
            kmeans = faiss.Kmeans(d, n_clusters, niter=20, verbose=False, gpu=False)
            kmeans.train(data_normalized)
            _, labels = kmeans.index.search(data_normalized, 1)
            labels = labels.flatten()
            
            info = {
                'method': 'faiss_kmeans',
                'n_clusters': n_clusters,
                'N': N,
            }
            
            return labels, info
            
        except ImportError:
            # Fallback to sklearn if FAISS not available
            pass
    
    # For smaller datasets or if FAISS fails, use sklearn
    from sklearn.cluster import DBSCAN, AgglomerativeClustering
    
    if method == 'dbscan' or (method == 'faiss' and N <= 10000):
        # Use DBSCAN for smaller datasets
        eps = np.std(log_scores) * 0.5
        min_samples = max(2, int(N * 0.01))
        
        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clustering.fit_predict(log_scores)
        
        info = {
            'method': 'dbscan',
            'eps': eps,
            'min_samples': min_samples,
            'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
            'N': N,
        }
        
    elif method == 'hierarchical':
        # Use hierarchical clustering (only for N < 10000)
        if N > 10000:
            raise ValueError("Hierarchical clustering not recommended for N > 10000 (memory constraint)")
        
        if n_clusters is None:
            n_clusters = max(2, int(np.sqrt(N / 2)))
        
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        labels = clustering.fit_predict(log_scores)
        
        info = {
            'method': 'hierarchical',
            'n_clusters': n_clusters,
            'N': N,
        }
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return labels, info

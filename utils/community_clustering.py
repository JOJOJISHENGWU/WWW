"""
Community clustering utility (review package version).

Given adjacency matrix A, build normalized Laplacian and run spectral clustering
with sklearn KMeans on first k eigenvectors.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from sklearn.cluster import KMeans


def normalized_laplacian(adj: np.ndarray) -> np.ndarray:
    deg = np.sum(adj, axis=1)
    deg = np.clip(deg, 1e-8, None)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
    i = np.eye(adj.shape[0], dtype=np.float64)
    return i - d_inv_sqrt @ adj @ d_inv_sqrt


def spectral_cluster(adj: np.ndarray, num_communities: int, seed: int = 42) -> np.ndarray:
    lap = normalized_laplacian(adj.astype(np.float64))
    eigvals, eigvecs = np.linalg.eigh(lap)
    feat = eigvecs[:, :num_communities]

    km = KMeans(n_clusters=num_communities, random_state=seed, n_init=10)
    labels = km.fit_predict(feat)
    return labels.astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build community assignment by spectral clustering")
    parser.add_argument("--adj", type=str, required=True, help="Path to adj.npy")
    parser.add_argument("--out", type=str, required=True, help="Path to output community.npy")
    parser.add_argument("--num_communities", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    adj = np.load(args.adj)
    labels = spectral_cluster(adj, num_communities=args.num_communities, seed=args.seed)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, labels)

    unique, counts = np.unique(labels, return_counts=True)
    dist = {int(u): int(c) for u, c in zip(unique, counts)}
    print(f"Saved community labels to: {args.out}")
    print(f"Distribution: {dist}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from collections import deque
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from load_datasets import get_blobs, get_wine, calculate_distances
from mst import KruskalMST, Edge
# -----------------------------
# MST clustering
# -----------------------------
def mst_cut_clustering(mst_edges: List[Edge], n: int, k: int) -> np.ndarray:
    """
    Remove (k-1) largest edges from MST and return cluster labels in 1..k for each node.
    """
    if k < 1:
        raise ValueError("k must be >= 1.")
    if n == 0:
        return np.array([], dtype=int)
    if k > n:
        raise ValueError(f"k must be <= n. Got k={k}, n={n}.")

    if k == 1:
        return np.ones(n, dtype=int)

    # Sort MST edges by weight descending and "cut" the top (k-1)
    mst_sorted = sorted(mst_edges, key=lambda e: e.w, reverse=True)
    cut_edges = set((e.u, e.v) if e.u < e.v else (e.v, e.u) for e in mst_sorted[: k - 1])

    # Build adjacency of remaining forest
    adj = [[] for _ in range(n)]
    for e in mst_edges:
        uv = (e.u, e.v) if e.u < e.v else (e.v, e.u)
        if uv in cut_edges:
            continue
        adj[e.u].append(e.v)
        adj[e.v].append(e.u)

    # Connected components -> clusters
    labels = np.zeros(n, dtype=int)
    current_label = 0

    for start in range(n):
        if labels[start] != 0:
            continue
        current_label += 1
        q = deque([start])
        labels[start] = current_label
        while q:
            u = q.popleft()
            for v in adj[u]:
                if labels[v] == 0:
                    labels[v] = current_label
                    q.append(v)

    # In a tree cut into k parts, we should get exactly k components
    # But due to edge ties / numeric quirks, we just normalize to 1..num_components
    return labels


# -----------------------------
# Plotting
# -----------------------------
def plot_2d_points(X2: np.ndarray, labels: np.ndarray, title: str) -> None:
    plt.figure()
    plt.scatter(X2[:, 0], X2[:, 1], c=labels, s=50)
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.savefig(f"figs/{title}.png")


def plot_dataset(X: np.ndarray, labels: np.ndarray, dataset_name: str) -> None:
    if X.shape[1] == 2:
        plot_2d_points(X, labels, f"MST clustering ({dataset_name})")
        
    else:
        # PCA to 2D for visualization
        pca = PCA(n_components=2, random_state=42)
        X2 = pca.fit_transform(X)
        plot_2d_points(X2, labels, f"MST clustering ({dataset_name}) - PCA2D (var={pca.explained_variance_ratio_.sum():.2f})")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["blobs", "wine"], required=True)
    ap.add_argument("--k", type=int, required=True, help="Number of clusters to form by cutting k-1 MST edges")
    ap.add_argument("--n", type=int, default=40, help="For blobs only: number of points (default 40)")
    args = ap.parse_args()

    # Load data
    if args.dataset == "blobs":
        X, _ = get_blobs(n=args.n)
    else:
        X = get_wine()

    # Distances for complete graph
    D = calculate_distances(X)

    # MST via Kruskal
    mst_builder = KruskalMST()
    mst_edges, total = mst_builder.build_mst(D)

    # Cluster by cutting k-1 largest MST edges
    labels = mst_cut_clustering(mst_edges, n=X.shape[0], k=args.k)

    # Print which edges were cut (useful for report/debug)
    mst_sorted = sorted(mst_edges, key=lambda e: e.w, reverse=True)
    cut = mst_sorted[: max(0, args.k - 1)]
    print(f"MST total weight: {total:.6f}")
    print(f"Cutting {len(cut)} edges (largest weights):")
    for e in cut:
        print(f"  ({e.u}, {e.v}) w={e.w:.6f}")

    # Plot
    plot_dataset(X, labels, args.dataset)


if __name__ == "__main__":
    main()
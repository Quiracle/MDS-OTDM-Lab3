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
def plot_2d_points(X2: np.ndarray, labels: np.ndarray, title: str, mst_edges: List[Edge], cut_edges_set: set) -> None:
    plt.figure()
    
    # Plot edges
    for e in mst_edges:
        p1 = X2[e.u]
        p2 = X2[e.v]
        
        # Check if this edge is cut
        key = (e.u, e.v) if e.u < e.v else (e.v, e.u)
        if key in cut_edges_set:
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], c='red', linestyle='--', linewidth=1, alpha=0.8)
        else:
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], c='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    plt.scatter(X2[:, 0], X2[:, 1], c=labels, s=50, zorder=3)
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.savefig(f"figs/{title}.png")


def plot_dataset(X: np.ndarray, labels: np.ndarray, dataset_name: str, mst_edges: List[Edge], cut_edges: List[Edge]) -> None:
    # Convert cut_edges to a set of tuples for faster lookup
    cut_edges_set = set()
    for e in cut_edges:
        cut_edges_set.add((e.u, e.v) if e.u < e.v else (e.v, e.u))

    if X.shape[1] == 2:
        plot_2d_points(X, labels, f"MST clustering ({dataset_name})", mst_edges, cut_edges_set)
        
    else:
        # PCA to 2D for visualization
        pca = PCA(n_components=2, random_state=42)
        X2 = pca.fit_transform(X)
        plot_2d_points(X2, labels, f"MST clustering ({dataset_name}) - PCA2D (var={pca.explained_variance_ratio_.sum():.2f})", mst_edges, cut_edges_set)


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
    
    # Calculate objective function: sum of distances of remaining edges
    cut_set = set(cut)
    remaining_weight = 0.0
    for e in mst_edges:
        if e not in cut_set:
            # Explicitly look up distance in D as requested
            dist = D[e.u, e.v]
            remaining_weight += dist
            
            
    print(f"Total weight of remaining edges (Objective Function): {remaining_weight:.6f}")

    # -----------------------------
    # Calculate Cluster Centers and Costs
    # -----------------------------
    print("\nCluster Centers and Costs:")
    total_cluster_cost = 0.0
    
    # Get unique labels (clusters), usually 1..k
    unique_labels = sorted(set(labels))
    
    for label in unique_labels:
        # Indices of points in this cluster
        indices = np.where(labels == label)[0]
        
        if len(indices) == 0:
            continue
            
        # 1. Calculate geometric centroid
        cluster_points = X[indices]
        centroid = np.mean(cluster_points, axis=0)
        
        # 2. Find the center node (closest to centroid)
        # Calculate distances from each point in cluster to centroid
        # We can use np.linalg.norm for Euclidean distance
        dists_to_centroid = np.linalg.norm(cluster_points - centroid, axis=1)
        best_local_idx = np.argmin(dists_to_centroid)
        center_node = indices[best_local_idx]
        
        # 3. Calculate Cost: Sum of D[center_node, other_node] for all nodes in cluster
        # Using the pre-calculated distance matrix D
        cost = 0.0
        for idx in indices:
            cost += D[center_node, idx]
            
        total_cluster_cost += cost
        print(f"  Cluster {label}: Center Node {center_node}, Cost {cost:.6f}")

    print(f"Total Cluster Cost (sum of distances to centers): {total_cluster_cost:.6f}")

    # Plot
    plot_dataset(X, labels, args.dataset, mst_edges, cut)


if __name__ == "__main__":
    main()
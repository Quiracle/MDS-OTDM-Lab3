#!/usr/bin/env python3
import numpy as np
import argparse
from load_datasets import get_blobs, get_wine, calculate_distances
from mst import KruskalMST, Edge
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import defaultdict, deque


def cut_mst_and_get_labels(mst_edges, n, k):
    edges_sorted = sorted(mst_edges, key=lambda e: e.w, reverse=True)
    edges_kept = edges_sorted[k-1:]

    adj = defaultdict(list)
    for e in edges_kept:
        adj[e.u].append(e.v)
        adj[e.v].append(e.u)

    labels = -np.ones(n, dtype=int)
    current_label = 0

    for i in range(n):
        if labels[i] != -1:
            continue
        queue = deque([i])
        labels[i] = current_label
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if labels[v] == -1:
                    labels[v] = current_label
                    queue.append(v)
        current_label += 1

    return labels


def medoid_cost_from_labels(D: np.ndarray, labels: np.ndarray) -> float:
    total = 0.0
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        subD = D[np.ix_(idx, idx)]
        costs = subD.sum(axis=0)
        medoid_local = np.argmin(costs)
        medoid = idx[medoid_local]
        total += D[idx, medoid].sum()
    return float(total)


def run(dataset: str, k: int, plot: bool):
    if dataset == "blobs":
        A, _ = get_blobs()
    elif dataset == "wine":
        A = get_wine()
    else:
        raise ValueError("dataset must be 'blobs' or 'wine'")

    D = calculate_distances(A)

    mst_builder = KruskalMST()
    mst_edges, mst_weight = mst_builder.build_mst(D)

    print(f"MST total edge weight: {mst_weight:.6f}")

    labels = cut_mst_and_get_labels(mst_edges, A.shape[0], k)

    z_mst = medoid_cost_from_labels(D, labels)
    print(f"MST clustering objective (medoid assignment cost): {z_mst:.6f}")

    if plot:
        if A.shape[1] > 2:
            X_plot = PCA(n_components=2).fit_transform(A)
        else:
            X_plot = A

        plt.figure(figsize=(6, 5))
        plt.scatter(X_plot[:, 0], X_plot[:, 1], c=labels, cmap="tab10", s=40)
        plt.title(f"MST clustering on {dataset}")
        plt.xlabel("PC1" if A.shape[1] > 2 else "x1")
        plt.ylabel("PC2" if A.shape[1] > 2 else "x2")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["blobs", "wine"], required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    run(args.dataset, args.k, args.plot)

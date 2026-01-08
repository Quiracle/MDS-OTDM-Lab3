#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from load_datasets import get_blobs, get_wine

# use regex
HEADER_RE = re.compile(r"^\s*:\s+(.*?)\s*:=\s*$")
ROW_RE = re.compile(r"^\s*(\d+)\s+(.*?)\s*$")

def _parse_header_cols(line: str) -> List[int]:
    m = HEADER_RE.match(line)
    if not m:
        return []
    return [int(tok) for tok in m.group(1).split()]

def parse_ampl_x_matrix(path: str) -> Dict[Tuple[int, int], float]:
    x: Dict[Tuple[int, int], float] = {}
    current_cols: List[int] = []
    in_matrix = False

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")

            if not in_matrix:
                if "x [*,*]" in line:
                    in_matrix = True
                continue

            if line.strip() == ";":
                break
            if not line.strip():
                continue

            if line.lstrip().startswith(":"):
                current_cols = _parse_header_cols(line)
                continue

            m = ROW_RE.match(line)
            if not m or not current_cols:
                continue

            i = int(m.group(1))
            vals = m.group(2).split()

            if len(vals) < len(current_cols):
                raise ValueError(
                    f"Row {i} has {len(vals)} values but header has {len(current_cols)} columns.\n"
                    f"Line: {line}"
                )

            for j, v_str in zip(current_cols, vals[:len(current_cols)]):
                x[(i, j)] = float(v_str)

    if not x:
        raise ValueError(f"No x[i,j] entries parsed from {path}.")
    return x

def assignments_from_x(x: Dict[Tuple[int, int], float], threshold: float = 0.5) -> Dict[int, int]:
    by_i: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for (i, j), v in x.items():
        by_i[i].append((j, v))

    assign: Dict[int, int] = {}
    for i, pairs in by_i.items():
        ones = [j for (j, v) in pairs if v > threshold]
        if len(ones) == 1:
            assign[i] = ones[0]
        elif len(ones) > 1:
            assign[i] = min(ones)  # deterministic tie-break
        else:
            assign[i] = max(pairs, key=lambda t: (t[1], -t[0]))[0]
    return assign

def compact_cluster_ids(assign_median: Dict[int, int]) -> Tuple[Dict[int, int], Dict[int, int]]:
    medians = sorted(set(assign_median.values()))
    median_to_cluster = {m: idx + 1 for idx, m in enumerate(medians)}
    point_cluster = {i: median_to_cluster[j] for i, j in assign_median.items()}
    return point_cluster, median_to_cluster


def plot_2d_points(X2: np.ndarray, cluster: np.ndarray, title: str, median_indices: List[int] = None):
    plt.figure()
    # Plot all points
    plt.scatter(X2[:, 0], X2[:, 1], c=cluster, s=50, cmap='viridis')
    
    # Highlight center points with a different color
    if median_indices is not None:
        # Convert 1-indexed median indices to 0-indexed
        center_idx = [m - 1 for m in median_indices]
        plt.scatter(X2[center_idx, 0], X2[center_idx, 1], 
                    c='red', s=150, marker='*', edgecolors='black', linewidths=1.5,
                    label='Centers', zorder=5)
        plt.legend()
    
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.savefig(f"figs/{title}.png")

def plot_wine_pca(X: np.ndarray, cluster: np.ndarray, title: str, median_indices: List[int] = None):
    """
    PCA reduce Wine to 2D for visualization.
    """
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X)
    plot_2d_points(X2, cluster, title + f" (PCA 2D, var={pca.explained_variance_ratio_.sum():.2f})", median_indices)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sol", required=True, help="AMPL solution file containing matrix display of x (e.g., blobs_sol.txt)")
    ap.add_argument("--dataset", choices=["blobs", "wine"], required=True, help="Which dataset was used to create this solution")
    ap.add_argument("--n", type=int, default=40, help="For blobs: number of points used when solving (default 40)")
    ap.add_argument("--threshold", type=float, default=0.5, help="Binary threshold for x (default 0.5)")
    args = ap.parse_args()

    if args.dataset == "blobs":
        X, _y = get_blobs(n=args.n)  # X shape (n,2)
    else:
        X = get_wine()              # X shape (178,13)

    x = parse_ampl_x_matrix(args.sol)
    assign = assignments_from_x(x, threshold=args.threshold)
    point_cluster, median_to_cluster = compact_cluster_ids(assign)

    m = X.shape[0]
    cluster = np.empty(m, dtype=int)
    for i in range(1, m + 1):
        if i not in point_cluster:
            raise ValueError(f"Missing assignment for point i={i}. Did AMPL output include all rows?")
        cluster[i - 1] = point_cluster[i]

    median_indices = list(median_to_cluster.keys())
    
    print("Medians used (median_j -> cluster_id):")
    for j, cid in sorted(median_to_cluster.items(), key=lambda t: t[1]):
        print(f"  {j} -> {cid}")
    
    print("\nCenter coordinates:")
    for j, cid in sorted(median_to_cluster.items(), key=lambda t: t[1]):
        center_coords = X[j - 1]
        print(f"  Cluster {cid} (point {j}): {center_coords}")

    if args.dataset == "blobs":
        if X.shape[1] != 2:
            raise ValueError("Blobs expected to be 2D (n_features=2).")
        plot_2d_points(X, cluster, "Blobs clustered by AMPL x[i,j]", median_indices)
    else:
        plot_wine_pca(X, cluster, "Wine clustered by AMPL x[i,j]", median_indices)

if __name__ == "__main__":
    main()

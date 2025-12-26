from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass(frozen=True)
class Edge:
    u: int
    v: int
    w: float


class UnionFind:
    """Disjoint-set union (Union-Find) with path compression + union by rank."""
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


class KruskalMST:
    """
    Build a Minimum Spanning Tree (MST) from a symmetric distance matrix D,
    interpreted as a complete undirected weighted graph.

    Nodes are 0..n-1 (matrix indices). If you use 1..m in AMPL, convert as needed.

    Notes:
    - Requires D to be square and symmetric (within tolerance).
    - Assumes nonnegative weights (typical for distances).
    """

    def __init__(self, sym_tol: float = 1e-9, ignore_diagonal: bool = True):
        self.sym_tol = sym_tol
        self.ignore_diagonal = ignore_diagonal

    def _validate(self, D: np.ndarray) -> None:
        if not isinstance(D, np.ndarray):
            raise TypeError("D must be a numpy.ndarray.")
        if D.ndim != 2 or D.shape[0] != D.shape[1]:
            raise ValueError(f"D must be square. Got shape={D.shape}.")
        if not np.all(np.isfinite(D)):
            raise ValueError("D contains NaN or inf values.")
        if not np.allclose(D, D.T, atol=self.sym_tol, rtol=0):
            max_asym = np.max(np.abs(D - D.T))
            raise ValueError(f"D must be symmetric. max|D-D.T|={max_asym:g} exceeds tol={self.sym_tol:g}.")

    def edges_from_distance_matrix(self, D: np.ndarray) -> List[Edge]:
        """
        Convert a symmetric distance matrix into a unique undirected edge list.
        Only uses upper triangle (u < v).
        """
        self._validate(D)
        n = D.shape[0]
        edges: List[Edge] = []

        for u in range(n):
            start_v = u + 1
            for v in range(start_v, n):
                w = float(D[u, v])
                edges.append(Edge(u=u, v=v, w=w))

        return edges

    def build_mst(self, D: np.ndarray) -> Tuple[List[Edge], float]:
        """
        Run Kruskal's algorithm.
        Returns (mst_edges, total_weight).
        """
        edges = self.edges_from_distance_matrix(D)
        edges.sort(key=lambda e: e.w)

        n = D.shape[0]
        uf = UnionFind(n)
        mst: List[Edge] = []
        total = 0.0

        for e in edges:
            if uf.union(e.u, e.v):
                mst.append(e)
                total += e.w
                if len(mst) == n - 1:
                    break

        if len(mst) != n - 1:
            # For a complete graph this shouldn't happen unless n==0/1 or data invalid.
            raise RuntimeError(f"MST incomplete: got {len(mst)} edges for n={n} nodes.")

        return mst, total



# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    import load_datasets

    X_blobs, Y_blobs = load_datasets.get_blobs()
    D = load_datasets.calculate_distances(X_blobs)

    mst_builder = KruskalMST()
    mst_edges, total = mst_builder.build_mst(D)

    print("MST edges (u, v, w):")
    for e in mst_edges:
        print(e.u, e.v, e.w)
    print("Total weight:", total)

    adj = mst_builder.mst_adjacency_list(n=D.shape[0], mst_edges=mst_edges)
    print("Adjacency list:", adj)

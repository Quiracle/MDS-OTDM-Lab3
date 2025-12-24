import numpy as np
from sklearn.datasets import load_wine, make_blobs
from scipy.spatial.distance import cdist

def get_blobs(n = 40):
    # Generate synthetic 2D blobs
    return make_blobs(
        n_samples=n,
        n_features=2,
        centers=4,
        cluster_std=0.8,
        random_state=42
    )

def get_wine():
    wine = load_wine()
    return wine.data 

def reduce_ds(dataset, n:int = 40):
    np.random.seed(42)
    idx = np.random.choice(dataset.shape[0], n, replace=False)
    return dataset[idx]

def calculate_distances(dataset):
    return cdist(dataset, dataset, metric="euclidean") 

def export_to_ampl_dat(D, filename, k):
    m = D.shape[0]
    with open(filename, "w") as f:
        f.write(f"param k := {k};\n\n")
        f.write(f"set I := {' '.join(str(i+1) for i in range(m))};\n\n")
        f.write("param d :\n")
        f.write("    " + " ".join(str(i+1) for i in range(m)) + " :=\n")
        for i in range(m):
            row = " ".join(f"{D[i,j]:.4f}" for j in range(m))
            f.write(f"{i+1}   {row}\n")
        f.write(";\n")

if __name__ == "__main__":

    X_blobs, Y_blobs = get_blobs()
    wine = get_wine()
    wine_sm = reduce_ds(wine)
    D_blobs = calculate_distances(X_blobs)
    D_wine = calculate_distances(wine)
    D_wine_sm = calculate_distances(wine_sm)

    export_to_ampl_dat(D_blobs, "blobs.dat", k=4)
    export_to_ampl_dat(D_wine, "wine.dat", k=3)
    export_to_ampl_dat(D_wine_sm, "wine_sm.dat", k=3)

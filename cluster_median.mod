set I;
param k integer >= 1;       # number of clusters to build
param d {I, I} >= 0;        # pairwise distances dij

var x {I, I} binary;        # x[i,j] = 1 if point i assigned to cluster with median j

minimize Total_Distance:
    sum {i in I, j in I} d[i,j] * x[i,j];

subject to AssignEachPoint {i in I}:
    sum {j in I} x[i,j] = 1;

subject to ExactlyKClusters:
    sum {j in I} x[j,j] = k;

subject to OpenClusterIfAssigned {i in I, j in I}:
    x[j,j] >= x[i,j];
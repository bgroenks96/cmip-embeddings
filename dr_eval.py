import numpy as np
import scipy.stats as stats
import itertools as it
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_distances
from sklearn.utils.graph_shortest_path import graph_shortest_path
from scipy.stats import pearsonr, spearmanr

def trust_cont_score(X, X_map, k=10, alpha=0.5, impute_strategy='median'):
    """
    Computes the "trustworthiness" and "continuity" [1] of X_map with respect to X.
    This is a port and extension of the implementation provided by Van der Maaten [2].
    
    Parameters:
    X     : the data in its original representation
    X_map : the lower dimensional representation of the data to be evaluated
    k     : parameter that determines the size of the neighborhood for the T&C measure
    alpha : mixing parameter in [0,1] that determines the weight given to trustworthiness vs. continuity; higher values will give more
            weight to trustworthiness, lower values to continuity.
    
    [1] Kaski S, Nikkilä J, Oja M, Venna J, Törönen P, Castrén E. Trustworthiness and metrics in visualizing similarity of gene expression. BMC bioinformatics. 2003 Dec;4(1):48.
    [2] Maaten L. Learning a parametric embedding by preserving local structure. InArtificial Intelligence and Statistics 2009 Apr 15 (pp. 384-391).
    """
    # Compute pairwise distance matrices
    D_h = pairwise_distances(X, X, metric='euclidean')
    D_l = pairwise_distances(X_map, X_map, metric='euclidean')
    # Compute neighborhood indices
    ind_h = np.argsort(D_h, axis=1)
    ind_l = np.argsort(D_l, axis=1)
    # Compute trustworthiness
    N = X.shape[0]
    T = 0
    C = 0
    t_ranks = np.zeros((k, 1))
    c_ranks = np.zeros((k, 1))
    for i in range(N):
        for j in range(k):
            t_ranks[j] = np.where(ind_h[i,:] == ind_l[i, j+1])
            c_ranks[j] = np.where(ind_l[i,:] == ind_h[i, j+1])
        t_ranks -= k
        c_ranks -= k
        T += np.sum(t_ranks[np.where(t_ranks > 0)])
        C += np.sum(c_ranks[np.where(c_ranks > 0)])
    S = (2 / (N * k * (2 * N - 3 * k - 1)))
    T = 1.0 - S*T
    C = 1.0 - S*C
    return alpha*T + (1.0-alpha)*C

def sammon_stress(X, X_m, impute_strategy='median'):
    X = Imputer(strategy=impute_strategy).fit_transform(X)
    Dx = pairwise_distances(X, X, metric='euclidean')
    Dy = pairwise_distances(X_m, X_m, metric='euclidean')
    # Sammon Stress computes sums over indices where i < j
    # We can interpet this as being the upper triangle of each matrix, from the k=1 diagonal
    Dx_ut = np.triu(Dx, k=1)
    Dy_ut = np.triu(Dy, k=1)
    # Compute Sammon Stress, S
    S = (1 / np.sum(Dx_ut))*np.sum(np.square(Dx_ut - Dy_ut) / (Dx_ut + np.ones(Dx.shape)))
    return S
    
    
def residual_variance(X, X_m, n_neighbors=20):
    kng_h = kneighbors_graph(X, n_neighbors=n_neighbors, mode='distance', n_jobs=mp.cpu_count()).toarray()
    D_h = graph_shortest_path(kng_h, method='D', directed=False)
    #D_h = pairwise_distances(X, X, metric='euclidean')
    #D_l = kneighbors_graph(X_m, n_neighbors=50, mode='distance').toarray()
    D_l = pairwise_distances(X_m, X_m, metric='euclidean')
    r,_ = spearmanr(D_h.flatten(), D_l.flatten())
    return 1 - r**2.0
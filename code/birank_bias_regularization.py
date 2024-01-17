import networkx as nx
import pandas as pd
import numpy as np
import scipy
import scipy.sparse as spa

def birank_with_regularization(W, gamma=0.01, normalizer='BiRank', alpha=0.85, beta=0.85, max_iter=200, tol=1.0e-4, verbose=False):
    """
    BiRank with user activity regularization integrated into the iterative process.

    Inputs:
        W::scipy's sparse matrix: Adjacency matrix of the bipartite network
        gamma::float: Regularization strength
        normalizer::string: Choose which normalizer to use
        alpha, beta::float: Damping factors for the rows and columns
        max_iter::int: Maximum iteration times
        tol::float: Error tolerance to check convergence
        verbose::boolean: If print iteration information

    Outputs:
        d, p::numpy.ndarray: The BiRank for rows and columns with regularization
    """

    W = W.astype('float', copy=False)
    WT = W.T

    Kd = scipy.array(W.sum(axis=1)).flatten()
    Kp = scipy.array(W.sum(axis=0)).flatten()

    # Avoid division by zero
    Kd[np.where(Kd == 0)] += 1
    Kp[np.where(Kp == 0)] += 1

    Kd_ = spa.diags(1/Kd)
    Kp_ = spa.diags(1/Kp)

    if normalizer == 'HITS':
        Sp = WT
        Sd = W
    elif normalizer == 'CoHITS':
        Sp = WT.dot(Kd_)
        Sd = W.dot(Kp_)
    elif normalizer == 'BGRM':
        Sp = Kp_.dot(WT).dot(Kd_)
        Sd = Sp.T
    elif normalizer == 'BiRank':
        Kd_bi = spa.diags(1 / scipy.sqrt(Kd))
        Kp_bi = spa.diags(1 / scipy.sqrt(Kp))
        Sp = Kp_bi.dot(WT).dot(Kd_bi)
        Sd = Sp.T

    d0 = np.repeat(1 / W.shape[0], W.shape[0])
    d_last = d0.copy()
    p0 = np.repeat(1 / W.shape[1], W.shape[1])
    p_last = p0.copy()

    # Calculate user activity as the degree of each user divided by the sum of degrees of all users
    total_degree = Kd.sum()
    user_activity = Kd / total_degree
    # Regularization factor (1e-6 is added to avoid division by zero)
    r = gamma / (user_activity + 1e-6)

    for i in range(max_iter):
        d = (alpha * Sd.dot(p_last) + (1 - alpha) * d0)* r
        p = alpha * Sp.dot(d) + (1 - alpha) * p0

        # Re-normalize after applying regularization
        d /= d.sum()
        p /= p.sum()

        err_d = np.absolute(d - d_last).sum()
        err_p = np.absolute(p - p_last).sum()

        if verbose:
            print(f"Iteration: {i}; Top error: {err_d}; Bottom error: {err_p}")

        if err_d < tol and err_p < tol:
            break

        d_last = d
        p_last = p

    return d, p

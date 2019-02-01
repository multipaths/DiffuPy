# -*- coding: utf-8 -*-

"""Implementation of the diffupy Kernel."""

import sys
from math import pi

import logging
import networkx as nx
import numpy as np
import scipy as sp

logger = logging.getLogger()


def get_laplacian(graph, normalized=False):
    """"""
    if nx.is_directed(graph):
        raise ValueError('Graph must be undirected')

    if not normalized:
        L = nx.laplacian_matrix(graph).toarray()
    else:
        L = nx.normalized_laplacian_matrix(graph).toarray()

    return L


def set_diagonal_matrix(matrix, d):
    """"""
    for j, row in enumerate(matrix):
        for i, x in enumerate(row):
            if i == j:
                matrix[j][i] = d[i]
            else:
                matrix[j][i] = x
    return matrix


def commute_time_kernel(graph, normalized=False):
    """Computes the conmute-time kernel, which is the expected time of going back and forth between a couple of nodes.
    If the network is connected, then the commute time kernel will be totally dense, therefore reflecting global
    properties of the network. For further details, see [Yen, 2007]. This kernel can be computed using both the
    unnormalised and normalised graph Laplacian."""

    # Apply pseudo-inverse (moore-penrose) of laplacian matrix
    return np.linalg.pinv(get_laplacian(graph, normalized))


def diffusion_kernel(graph, sigma2=1, normalized=True):
    """"""
    EL = -sigma2 / 2 * get_laplacian(graph, normalized)
    return sp.linalg.expm(EL)


def inverse_cosine_kernel(graph):
    """Computes the inverse cosine kernel, which is based on a cosine transform on the spectrum of the normalized Laplacian
    matrix. Quoting [Smola, 2003]: the inverse cosine kernel treats lower complexity functions almost equally, with a
    significant reduction in the upper end of the spectrum. This kernel is computed using the normalised graph Laplacian."""
    # Decompose matrix (Singular Value Decomposition)
    U, S, _ = np.linalg.svd(get_laplacian(graph, normalized=True) * (pi / 4))

    return np.matmul(np.matmul(U, np.diag(np.cos(S))), np.transpose(U))


def p_step_kernel(graph, a=2, p=5):
    """Computes the p-step random walk kernel. This kernel is more focused on local properties of the nodes, because
    random walks are limited in terms of length. Therefore, if p is small, only a fraction of the values K(x1,x2) will
    be non-null if the network is sparse [Smola, 2003].
    The parameter a is a regularising term that is summed to the spectrum of the normalised Laplacian matrix, and has
    to be 2 or greater. The p-step kernels can be cheaper to compute and have been successful in biological tasks,
    see the benchmark in [Valentini, 2014].
    """
    minusL = -get_laplacian(graph, normalized=True)

    # Not optimal but kept for clarity
    # here we restrict to the normalised version, as the eigenvalues are
    # between 0 and 2 -> restriction a >= 2
    if a < 2:
        sys.exit('Eigenvalues must be between 0 and 2')
    if p < 0:
        sys.exit('p must be greater than 0')

    M = set_diagonal_matrix(minusL, [x + a for x in np.diag(minusL)])

    if p == 1: return M

    return np.linalg.matrix_power(M, p)


def regularised_laplacian_kernel(G, sigma2=1, add_diag=1, normalized=False):
    """Computes the regularised Laplacian kernel, which is a standard in biological networks.
    The regularised Laplacian kernel arises in numerous situations, such as the finite difference formulation of the
    diffusion equation and in Gaussian process estimation. Sticking to the heat diffusion model, this function allows
    to control the constant terms summed to the diagonal through add_diag, i.e. the strength of the leaking in each node.
    If a node has diagonal term of 0, it is not allowed to disperse heat. The larger the diagonal term of a node, the
    stronger the first order heat dispersion in it, provided that it is positive. Every connected component in the graph
    should be able to disperse heat, i.e. have at least a node i with add_diag[i] > 0. If this is not the case, the result
    diverges. More details on the parameters can be found in [Smola, 2003].
    This kernel can be computed using both the unnormalised and normalised graph Laplacian.
    """
    L = get_laplacian(G, normalized)
    RL = set_diagonal_matrix(sigma2 * L, [x + add_diag for x in np.diag(L)])

    return np.linalg.inv(RL)

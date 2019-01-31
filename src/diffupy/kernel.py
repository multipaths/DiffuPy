# -*- coding: utf-8 -*-

import logging
import os
import sys
from math import pi

import networkx as nx
import numpy as np
import scipy as sp

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.debug("test")

dir_path = os.path.dirname(os.path.realpath('__file__'))


def get_laplacian(G, normalized=False):
    if nx.is_directed(G):
        sys.exit('Graph must be undirected')

    if not normalized:
        L = nx.laplacian_matrix(G).toarray()
    else:
        L = nx.normalized_laplacian_matrix(G).toarray()

    return L


def set_diagonal_matrix(M, d):
    for j, row in enumerate(M):
        for i, x in enumerate(row):
            if i == j:
                M[j][i] = d[i]
            else:
                M[j][i] = x
    return M


def commute_time_kernel(G, normalized=False):
    """Computes the conmute-time kernel, which is the expected time of going back and forth between a couple of nodes. If the network is connected, then the commute time kernel will be totally dense, therefore reflecting global properties of the network. For further details, see [Yen, 2007]. This kernel can be computed using both the unnormalised and normalised graph Laplacian.."""
    # Apply pseudo-inverse (moore-penrose) of laplacian matrix
    return np.linalg.pinv(get_laplacian(G, normalized))


def diffusion_kernel(G, sigma2=1, normalized=True):
    EL = -sigma2 / 2 * get_laplacian(G, normalized)
    return sp.linalg.expm(EL)


def inverse_cosine_kernel(G):
    # Decompose matrix (Singular Value Decomposition)
    U, S, _ = np.linalg.svd(get_laplacian(G, normalized=True) * (pi / 4))

    return np.matmul(np.matmul(U, np.diag(np.cos(S))), np.transpose(U))


def p_step_kernel(G, a=2, p=5):
    minusL = -get_laplacian(G, normalized=True)

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
    L = get_laplacian(G, normalized)
    RL = sigma2 * L

    RL = set_diagonal_matrix(RL, [x + add_diag for x in np.diag(L)])

    print(RL)
    A = np.array(RL[:, :-1])
    b = np.array(RL[:, -1])

    print(A)
    print(b)

    return np.linalg.solve(A, b)

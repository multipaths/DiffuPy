# -*- coding: utf-8 -*-

"""Compute graph kernels."""

import sys
from math import pi

import logging

import networkx as nx
import numpy as np
import scipy as sp
from .matrix import LaplacianMatrix, Matrix
from .miscellaneous import set_diagonal_matrix

log = logging.getLogger(__name__)


def commute_time_kernel(graph: nx.Graph, normalized: bool = False) -> Matrix:
    """Computes the conmute-time kernel, which is the expected time of going back and forth between a couple of nodes.
    If the network is connected, then the commute time kernel will be totally dense, therefore reflecting global
    properties of the network. For further details, see [Yen, 2007]. This kernel can be computed using both the
    unnormalised and normalised graph Laplacian."""

    # Apply pseudo-inverse (moore-penrose) of laplacian matrix
    L = LaplacianMatrix(graph, normalized)
    L.mat = np.linalg.pinv(L.mat)
    return L


def diffusion_kernel(graph: nx.Graph, sigma2: float = 1, normalized: bool = True) -> Matrix:
    """"""
    L = LaplacianMatrix(graph, normalized)
    L.mat = sp.linalg.expm(-sigma2 / 2 * L.mat)
    return L


def inverse_cosine_kernel(graph: nx.Graph) -> Matrix:
    """Computes the inverse cosine kernel, which is based on a cosine transform on the spectrum of the normalized Laplacian
    matrix. Quoting [Smola, 2003]: the inverse cosine kernel treats lower complexity functions almost equally, with a
    significant reduction in the upper end of the spectrum. This kernel is computed using the normalised graph Laplacian."""
    # Decompose matrix (Singular Value Decomposition)
    L = LaplacianMatrix(graph, normalized=True)
    # Decompose matrix (Singular Value Decomposition)
    U, S, _ = np.linalg.svd(L.mat * (pi / 4))
    L.mat = np.matmul(np.matmul(U, np.diag(np.cos(S))), np.transpose(U))
    return L


def p_step_kernel(graph: nx.Graph, a: float = 2, p: float = 5) -> Matrix:
    """Computes the p-step random walk kernel. This kernel is more focused on local properties of the nodes, because
    random walks are limited in terms of length. Therefore, if p is small, only a fraction of the values K(x1,x2) will
    be non-null if the network is sparse [Smola, 2003].
    The parameter a is a regularising term that is summed to the spectrum of the normalised Laplacian matrix, and has
    to be 2 or greater. The p-step kernels can be cheaper to compute and have been successful in biological tasks,
    see the benchmark in [Valentini, 2014].
    """
    M = LaplacianMatrix(graph, normalized=True)
    M.mat = -M.mat

    # Not optimal but kept for clarity
    # here we restrict to the normalised version, as the eigenvalues are
    # between 0 and 2 -> restriction a >= 2
    if a < 2:
        sys.exit('Eigenvalues must be between 0 and 2')
    if p < 0:
        sys.exit('p must be greater than 0')

    M.mat = set_diagonal_matrix(M.mat, [x + a for x in np.diag(M.mat)])

    if p == 1: return M

    M.mat = np.linalg.matrix_power(M.mat, p)

    return M


def regularised_laplacian_kernel(graph: nx.Graph, sigma2: float = 1, add_diag: int = 1,
                                 normalized: bool = False) -> Matrix:
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
    RL = LaplacianMatrix(graph, normalized)
    RL.mat = np.linalg.inv(set_diagonal_matrix(sigma2 * RL.mat, [x + add_diag for x in np.diag(RL.mat)]))

    return RL

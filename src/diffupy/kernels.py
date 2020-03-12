# -*- coding: utf-8 -*-

"""Compute graph kernels."""

import logging
import sys
import time
from math import pi

import networkx as nx
import numpy as np
import scipy as sp

from .matrix import LaplacianMatrix, Matrix
from .utils import set_diagonal_matrix

log = logging.getLogger(__name__)

__all__ = [
    'diffusion_kernel',
    'commute_time_kernel',
    'inverse_cosine_kernel',
    'regularised_laplacian_kernel',
    'p_step_kernel',
]


def commute_time_kernel(graph: nx.Graph, normalized: bool = False) -> Matrix:
    """Compute the commute-time kernel, which is the expected time of going back and forth between a couple of nodes.

    If the network is connected, then the commuted time kernel will be totally dense, therefore reflecting global
    properties of the network. For further details, see [Yen, 2007]. This kernel can be computed using both the
    unnormalised and normalised graph Laplacian.

    :param graph: A graph
    :param normalized: Indicates if Laplacian transformation is normalized or not.
    :return: Laplacian representation of the graph.
    """
    # Apply pseudo-inverse (moore-penrose) of laplacian matrix
    laplacian = LaplacianMatrix(graph, normalized)
    laplacian.mat = np.linalg.pinv(laplacian.mat)

    return laplacian


def diffusion_kernel(graph: nx.Graph, sigma2: float = 1, normalized: bool = True) -> Matrix:
    """Compute the classical diffusion kernel that involves matrix exponentiation.

    It has a "bandwidth" parameter sigma^2 that controls the extent of the spreading.
    Quoting [Smola, 2003]:
    k(x1,x2) can be visualized as the quantity of some substance that would accumulate at
    vertex x2 after a given amount of time if we injected the substance at vertex x1 and let
    it diffuse through the graph along the edges.

    This kernel can be computed using both the unnormalised and normalised graph Laplacian.
    :param graph: A graph
    :param sigma2: Controls the extent of the spreading.
    :param normalized: Indicates if Laplacian transformation is normalized or not.
    :return: Laplacian representation of the graph
    """
    laplacian = LaplacianMatrix(graph, normalized)
    laplacian.mat = sp.linalg.expm(-sigma2 / 2 * laplacian.mat)

    return laplacian


def inverse_cosine_kernel(graph: nx.Graph) -> Matrix:
    """Compute the inverse cosine kernel, which is based on a cosine transform  on the spectrum of the normalized LM.

    Quoting [Smola, 2003]: the inverse cosine kernel treats lower complexity
    functions almost equally, with a significant reduction in the upper end of the spectrum.

    This kernel is computed using the normalised graph Laplacian.

    :param graph: A graph
    :return: Laplacian representation of the graph
    """
    # Decompose matrix (Singular Value Decomposition)
    laplacian = LaplacianMatrix(graph, normalized=True)
    # Decompose matrix (Singular Value Decomposition)
    u, s, _ = np.linalg.svd(laplacian.mat * (pi / 4))
    laplacian.mat = np.matmul(np.matmul(u, np.diag(np.cos(s))), np.transpose(u))

    return laplacian


def p_step_kernel(graph: nx.Graph, a: int = 2, p: int = 5) -> Matrix:
    """Compute the inverse cosine kernel, which is based on a cosine transform on the spectrum of the normalized LM.

    This kernel is more focused on local properties of the nodes, because random walks
    are limited in terms of length. Therefore, if p is small, only a fraction of the values k(x1,x2)
    will be non-null if the network is sparse [Smola, 2003]. The parameter a is a regularising term
    that is summed to the spectrum of the normalised Laplacian matrix, and has to be 2 or greater.
    The p-step kernels can be cheaper to compute and have been successful in biological tasks, see the benchmark in
    [Valentini, 2014].

    :param graph: A graph
    :param a: regularising summed to the spectrum. Spectrum of the normalised Laplacian matrix.
    :param p: p-step kernels can be cheaper to compute and have been successful in biological tasks.
    :return: Laplacian repr'esentation of the graph.
    """
    laplacian = LaplacianMatrix(graph, normalized=True)
    laplacian.mat = -laplacian.mat

    # Not optimal but keep for clarity
    # here we restrict to the normalised version, as the eigenvalues are
    # between 0 and 2 -> restriction a >= 2
    if a < 2:
        sys.exit('Eigenvalues must be between 0 and 2')

    if p < 0:
        sys.exit('p must be greater than 0')

    laplacian.mat = set_diagonal_matrix(laplacian.mat, [x + a for x in np.diag(laplacian.mat)])

    if p == 1:
        return laplacian

    laplacian.mat = np.linalg.matrix_power(laplacian.mat, p)

    return laplacian


def regularised_laplacian_kernel(
        graph: nx.Graph,
        sigma2: float = 1,
        add_diag: int = 1,
        normalized: bool = False
) -> Matrix:
    """Compute the regularised Laplacian kernel, which is a standard in biological networks.

    The regularised Laplacian kernel arises in numerous situations, such as the finite difference formulation of the
    diffusion equation and in Gaussian process estimation. Sticking to the heat diffusion model, this function allows
    to control the constant terms summed to the diagonal through add_diag, i.e. the strength of the leaking in each node.
    If a node has diagonal term of 0, it is not allowed to disperse heat. The larger the diagonal term of a node, the
    stronger the first order heat dispersion in it, provided that it is positive. Every connected component in the graph
    should be able to disperse heat, i.e. have at least a node i with add_diag[i] > 0. If this is not the case, the result
    diverges. More details on the parameters can be found in [Smola, 2003].
    This kernel can be computed using both the unnormalised and normalised graph Laplacian.

    :param graph: A graph
    :param a: regularising summed to the spectrum. Spectrum of the normalised Laplacian matrix.
    :param p: p-step kernels can be cheaper to compute and have been successful in biological tasks.
    :return: Laplacian representation of the graph.

    """
    then = time.time()

    regularized_laplacian = LaplacianMatrix(graph, normalized)

    regularized_laplacian.mat = np.linalg.inv(
        set_diagonal_matrix(
            sigma2 * regularized_laplacian.mat,
            [x + add_diag
             for x in np.diag(regularized_laplacian.mat)
             ]
        )
    )
    now = time.time()
    log.info("The kernel generation took: ", now - then, " seconds")

    return regularized_laplacian

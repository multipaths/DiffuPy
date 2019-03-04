# -*- coding: utf-8 -*-

"""Diffuse scores on a network."""

import logging

import networkx as nx
import numpy as np

from .kernels import regularised_laplacian_kernel
from .matrix import Matrix
from .validate_inputs import _validate_scores, _validate_graph, _validate_K

logger = logging.getLogger()


def calculate_scores(
        col_ind: int,
        scores: np.array,
        diff: np.array,
        const_mean: np.array,
        const_var: np.array) -> float:
    """Helper function for diffuse_raw, which operate the z-scores calculation given a whole column of the score matrix.

    :param col_ind: background object for the diffusion
    :param scores: list of score matrices. For a single input with a single background, supply a list with a vector column
    :param diff: bool to indicate if z-scores be computed instead of raw scores
    :param const_mean: K optional matrix precomputed diffusion kernel
    :param const_var: K optional matrix precomputed diffusion kernel
    :return:  Calculated column z-score
    """
    col_in = scores[:, col_ind]
    col_raw = diff[:, col_ind]

    s1 = np.sum(col_in)
    s2 = np.sum(col_in ** 2)

    # means and vars depend on first and second moments
    # of the input. This should be valid for non-binary
    # inputs as well
    score_means = const_mean * s1
    score_vars = const_var * (len(scores) * s2 - s1 ** 2)

    return np.subtract(col_raw, score_means) / np.sqrt(score_vars)


def diffuse_raw(
        graph: nx.Graph,
        scores: Matrix,
        z: bool = False,
        K: Matrix = None,
        **karg) -> Matrix:
    """Computes the conmute-time kernel, which is the expected time of going back and forth between a couple of nodes.
        If the network is connected, then the commute time kernel will be totally dense, therefore reflecting global
        properties of the network. For further details, see [Yen, 2007]. This kernel can be computed using both the
        unnormalised and normalised graph Laplacian.

    :param graph: background object for the diffusion
    :param scores: list of score matrices. For a single input with a single background, supply a list with a vector column
    :param z-logical: bool to indicate if z-scores be computed instead of raw scores
    :param  K optional matrix precomputed diffusion kernel
    :return:  A list of scores, with the same length and dimensions as scores
    """
    # Sanity checks
    _validate_scores(scores)


    # Kernel matrix
    if K is None:
        _validate_graph(graph)
        logging.info('Kernel not supplied. Computing regularised Laplacian kernel ...')
        K = regularised_laplacian_kernel(graph, normalized=False)
        logging.info('Done')
    else:
        _validate_K(K)
        logging.info('Using supplied kernel matrix...')

    # Match indices
    scores = scores.match_rows(K)
    # TODO: Sparse
    # scores.mat <- methods::as(scores[[scores.name]], "sparseMatrix")

    # Compute scores

    n = len(scores.mat)
    K = K.mat

    # raw scores
    diff = np.matmul(K[:, :n], scores.mat)

    # Return base matrix if it is raw. Continue if we want z-scores
    if not z:
        return diff

    # If we want z-scores, must compute rowmeans and rowmeans2
    row_sums = np.array(
        [round(np.sum(row), 2)
         for row in K[:, :n]]
    )
    row_sums_2 = np.array(
        [np.sum(row)
         for row in K[:, :n] ** 2]
    )

    # Constant terms over columns
    const_mean = row_sums / n
    const_var = np.subtract(n * row_sums_2, row_sums ** 2) / ((n - 1) * (n ** 2))

    # Calculate z-scores iterating the score matrix columns, performing the operation with the whole column.
    return Matrix(
        np.transpose(
            [np.array(
                calculate_scores(
                    i,
                    scores.mat,
                    diff,
                    const_mean,
                    const_var
                )
            )
                for i in range(diff.shape[1])
            ]),
        scores.rows_labels,
        scores.cols_labels
    )

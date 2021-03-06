# -*- coding: utf-8 -*-

"""Diffuse scores on a network."""
import copy
import logging

import networkx as nx
import numpy as np

from .kernels import regularised_laplacian_kernel
from .matrix import Matrix
from .validate_input import _validate_graph, _validate_k

logger = logging.getLogger()

__all__ = [
    'diffuse_raw',
]


def _calculate_scores(
    col_ind: int,
    scores: np.array,
    raw_diff_scores: np.array,
    const_mean: np.array,
    const_var: np.array
) -> float:
    """Operate a column of the z-scores for the input score and the raw scores precomputation, helper function for diffuse_raw.

    :param col_ind: index of the column to operate
    :param scores: array of score matrices
    :param raw_diff_scores: pre-computed raw diffusion scores
    :param const_mean: pre-calculated constant mean over columns
    :param const_var: pre-calculated constant variance over columns
    :return: Calculated column z-score
    """
    col_in = scores[:, col_ind]
    col_raw = raw_diff_scores[:, col_ind]

    s1 = np.sum(col_in)
    s2 = np.sum(col_in ** 2)

    # means and vars depend on first and second moments
    # of the path. This should be valid for non-binary
    # inputs as well
    score_means = const_mean * s1
    score_vars = const_var * (len(scores) * s2 - s1 ** 2)

    return np.subtract(col_raw, score_means) / np.sqrt(score_vars)


def diffuse_raw(
    graph: nx.Graph,
    scores: Matrix,
    z: bool = False,
    k: Matrix = None,
) -> Matrix:
    """Compute the score diffusion procedure, given an initial state as a set of scores and a network to diffuse over.

    :param graph: background network
    :param scores: array of score matrices. For a single path with a single background, supply a list with a vector col
    :param z: bool to indicate if z-scores be computed instead of raw scores
    :param k: optional precomputed diffusion kernel matrix
    :return: A list of scores, with the same length and dimensions as scores
    """
    # Sanity checks
    # _validate_scores(scores)
    logging.info('Scores validated.')

    # Get the Kernel
    if k:
        kernel = copy.copy(k)
        _validate_k(kernel)
        logging.info('Using supplied kernel matrix...')
    elif graph:
        _validate_graph(graph)
        logging.info('Kernel not supplied. Computing regularised Laplacian kernel from the provided graph...')
        kernel = regularised_laplacian_kernel(graph, normalized=False)
        logging.info('Done')
    else:
        raise ValueError("No network, neither a graph or a kernel has been provided to run the diffusion.")

    # Match indices
    logging.info('Kernel validated scores.')

    scores = scores.match_rows(kernel)
    logging.info('Scores matched.')

    # TODO: Sparse
    # scores.mat <- methods::as(scores[[scores.name]], "sparseMatrix")

    # Compute scores

    n = len(scores.mat)
    kernel = kernel.mat

    # raw scores
    raw_scores_diff = np.matmul(kernel[:, :n], scores.mat)
    logging.info('Matrix product for raw scores preformed.')

    # Return base matrix if it is raw. Continue if we want z-scores.
    if not z:
        return Matrix(
            raw_scores_diff,
            rows_labels=scores.rows_labels,
            cols_labels=['output diffusion scores'],
            name=scores.name
        )

    logging.info('Normalization z-scores.')

    # If we want z-scores, must compute rowmeans and rowmeans2
    row_sums = np.array(
        [round(np.sum(row), 2)
         for row in kernel[:, :n]]
    )
    row_sums_2 = np.array(
        [np.sum(row)
         for row in kernel[:, :n] ** 2]
    )

    logging.info('Rowmeans and rowmeans2 computed.')

    # Constant terms over columns
    const_mean = row_sums / n
    const_var = np.subtract(n * row_sums_2, row_sums ** 2) / ((n - 1) * (n ** 2))

    # Calculate z-scores iterating the score matrix columns, performing the operation with the whole column.
    return Matrix(
        np.transpose(
            [np.array(
                _calculate_scores(
                    i,
                    scores.mat,
                    raw_scores_diff,
                    const_mean,
                    const_var
                )
            )
                for i in range(raw_scores_diff.shape[1])
            ]),
        rows_labels=scores.rows_labels,
        cols_labels=['output diffusion scores'],
        name=scores.name
    )

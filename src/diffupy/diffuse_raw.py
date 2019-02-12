# -*- coding: utf-8 -*-

"""Diffuse scores on a network."""

import numpy as np

from .checkers import check_scores, check_graph, check_K
from .kernel import regularised_laplacian_kernel
from .matrix import Matrix

import logging

logger = logging.getLogger()


def calculate_scores(col_ind, scores, diff, const_mean, const_var):
    """Helper function for diffuse_raw, which operates the z-score for each cell of the score matrix."""

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


def diffuse_raw(graph,
                scores,
                z=False,
                K=None,
                *argv):
    """Computes the conmute-time kernel, which is the expected time of going back and forth between a couple of nodes.
        If the network is connected, then the commute time kernel will be totally dense, therefore reflecting global
        properties of the network. For further details, see [Yen, 2007]. This kernel can be computed using both the
        unnormalised and normalised graph Laplacian"""

    # sanity checks
    check_scores(scores)

    # Kernel matrix
    if K is None:
        check_graph(graph)
        logging.info('Kernel not supplied. Computing regularised Laplacian kernel ...')
        K = regularised_laplacian_kernel(graph, normalized=False)
        logging.info('Done')
    else:
        check_K(K)
        logging.info('Using supplied kernel matrix...')

    # Compute scores

    # Match indices
    scores = scores.match_rows(K)

    # TODO: Sparse
    # scores.mat <- methods::as(scores[[scores.name]], "sparseMatrix")

    n = len(scores.mat)
    K = K.mat

    # raw scores
    diff = np.matmul(K[:, :n], scores.mat)

    # Return base matrix if it is raw
    # Continue if we want z-scores
    if not z:
        return diff

    # If we want z-scores, must compute rowmeans and rowmeans2
    row_sums = np.array([round(np.sum(row), 2) for row in K[:, :n]])
    row_sums_2 = np.array([np.sum(row) for row in K[:, :n] ** 2])

    # Constant terms over columns
    const_mean = row_sums / n
    const_var = np.subtract(n * row_sums_2, row_sums ** 2) / ((n - 1) * (n ** 2))

    return Matrix(np.transpose([np.array(calculate_scores(i,
                                                        scores.mat,
                                                        diff,
                                                        const_mean,
                                                        const_var
                                                          )
                                         )
                                for i in range(len(diff[0]))
                                ]),
                    scores.rows_labels,
                    scores.cols_labels
                )

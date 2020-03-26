# -*- coding: utf-8 -*-

"""This module provides a generalized function as an interface to interact with the different diffusion methods."""

import copy
import logging

import networkx as nx
import numpy as np

from .diffuse_raw import diffuse_raw
from .matrix import Matrix
from .utils import get_label_list_graph
from .validate_input import _validate_scores

log = logging.getLogger(__name__)

__all__ = [
    'diffuse',
]


def diffuse(
        input_scores,
        method: str = 'raw',
        graph: nx.Graph = None,
        **kwargs
) -> Matrix:
    """Run diffusion on a network given an input and a diffusion method.

    :param input_scores: score collection, supplied as n-dimensional array. Could be 1-dimensional (List) or n-dimensional (Matrix).
    :param method: Elected method ["raw", "ml", "gm", "ber_s", "ber_p", "mc", "z"]
    :param graph: A network as a graph. It could be optional if a Kernel is provided
    :param kwargs: Optional arguments:
                    - k: a  kernel [matrix] steaming from a graph, thus sparing the graph transformation process
                    - Other arguments which would differ depending on the chosen method
    :return: The diffused scores within the matrix transformation of the network, with the diffusion operation
             [k x input_vector] performed
    """
    # Sanity checks
    scores = copy.copy(input_scores)

    _validate_scores(scores)

    # Discern the sep of the provided network for its further treatment.
    if graph:
        format_network = 'graph'
    else:
        if 'k' not in kwargs:
            raise ValueError("Neither a graph 'graph' or a kernel 'k' has been provided.")
        format_network = 'kernel'

    if method == 'raw':
        return diffuse_raw(graph, scores, **kwargs)

    elif method == 'z':
        return diffuse_raw(graph, scores, z=True, **kwargs)

    elif method == 'ml':
        for score, i, j in scores.__iter__(get_labels=False, get_indices=True):
            if score not in [-1, 0, 1]:
                raise ValueError("Input scores must be binary.")
            if score == 0:
                scores.mat[i, j] = -1

        return diffuse_raw(graph, scores, **kwargs)

    elif method == 'gm':
        for score, i, j in scores.__iter__(get_labels=False, get_indices=True):
            if score not in [0, 1]:
                raise ValueError("Input scores must be binary.")
                # Have to match rownames with background
                # If the kernel is provided...

        if format_network == 'graph':
            names_ordered = get_label_list_graph(graph, 'name')
        elif format_network == 'kernel':
            names_ordered = kwargs['k'].rows_labels

        # If the graph is defined
        ids_nobkgd = set(names_ordered) - set(scores.rows_labels)

        n_tot = len(names_ordered)
        n_bkgd = scores.mat.shape[0]

        # Normalisation is performed for each column, as it depends on the number of positives and negatives.
        # n_pos and n_neg are vectors counting the number of positives and negatives in each column
        n_pos = np.sum(scores.mat, axis=0)
        n_neg = n_bkgd - n_pos

        # Biases
        p = (n_pos - n_neg) / n_tot

        for score, i, j in scores.__iter__(get_labels=False, get_indices=True):
            if score == 0:
                scores.mat[i, j] = -1

        # Add biases (each column has its bias)
        scores.row_bind(
            np.transpose(np.array(
                [np.repeat(
                    score,
                    n_tot - n_bkgd
                )
                    for score in p
                ])
            ),
            ids_nobkgd
        )

        return diffuse_raw(graph, scores, **kwargs)

    else:
        raise ValueError(f"Method not allowed {method}")

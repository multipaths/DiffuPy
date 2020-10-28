# -*- coding: utf-8 -*-

"""This module provides a generalized function as an interface to interact with the different diffusion methods."""

import copy
import logging
from typing import Union, Optional

import networkx as nx
import numpy as np
import pandas as pd

from diffupy.kernels import regularised_laplacian_kernel
from diffupy.process_input import process_map_and_format_input_data_for_diff

from .constants import *
from .diffuse_raw import diffuse_raw
from .matrix import Matrix
from .utils import get_label_list_graph
from .validate_input import _validate_scores

log = logging.getLogger(__name__)

__all__ = [
    'diffuse',
]

"""Map nodes from input to network"""


def run_diffusion(
        input_labels: Union[str, pd.DataFrame, list, dict, np.ndarray, Matrix],
        network: nx.Graph,
        method: Optional[str] = RAW,
        binarize: Optional[bool] = False,
        threshold: Optional[float] = None,
        absolute_value: Optional[bool] = False,
        p_value: Optional[float] = 0.05,
) -> Matrix:
    """Process and format miscellaneous data input and run diffusion over a provided graph network.

    :param input: (miscellaneous format) data input to be processed/formatted.
    :param network: Path to the network as a (NetworkX) graph or as a (diffuPy.Matrix) kernel.
    :param method:  Elected method ["raw", "ml", "gm", "ber_s", "ber_p", "mc", "z"]. By default 'raw'
    :param binarize: If logFC provided in dataset, convert logFC to binary. By default False
    :param threshold: Codify node labels by applying a threshold to logFC in input. By default None
    :param absolute_value: Codify node labels by applying threshold to | logFC | in input. By default False
    :param p_value: Statistical significance. By default 0.05
    """
    kernel = regularised_laplacian_kernel(network, normalized=False)

    formated_input_scores = process_map_and_format_input_data_for_diff(input_labels,
                                                                       kernel,
                                                                       method,
                                                                       binarize,
                                                                       absolute_value,
                                                                       p_value,
                                                                       threshold,
                                                                       )
    return diffuse(formated_input_scores, method, k=kernel)


def diffuse(
        input_scores: Matrix,
        method: Union[str, Callable] = RAW,
        graph: nx.Graph = None,
        **kwargs
) -> Matrix:
    """Run diffusion on a network given a formated input (matched with kernel Matrix) and a diffusion method.

    :param input_scores: score collection, supplied as n-dimensional array. Could be 1-dimensional (Vector) or n-dimensional (Matrix).
    :param method: Elected method ["raw", "ml", "gm", "ber_s", "ber_p", "mc", "z"] or custom method FUNCTION(network, scores, kargs).
    :param graph: A network as a graph. It could be optional if a Kernel is provided
    :param kwargs: Optional arguments:
                    - k: a  kernel [matrix] stemming from a graph, thus sparing the graph transformation process
                    - Other arguments which would differ depending on the chosen method
    :return: The diffused scores within the matrix transformation of the network, with the diffusion operation
             [k x input_vector] performed
    """
    # Sanity checks; create copy of input labels
    scores = copy.copy(input_scores)

    _validate_scores(scores)

    # Discern the sep of the provided network for its further treatment.
    if graph:
        format_network = 'graph'
    else:
        if 'k' not in kwargs:
            raise ValueError("Neither a graph 'graph' or a kernel 'k' has been provided.")
        format_network = 'kernel'

    # Allow custom method function.
    if callable(method):

        if format_network == 'graph':
            network = graph
        else:
            network = kwargs['k']

        if len(inspect.getfullargspec(method).args) == 1:
            return method(network)
        elif len(inspect.getfullargspec(method).args) == 2:
            return method(network, scores)
        elif len(inspect.getfullargspec(method).args) > 2:
            return method(graph, scores, **kwargs)
        else:
            raise ValueError(f"Method function arguments not allowed {type(method).__name__}")


    elif isinstance(method, str):

        if method == RAW:
            return diffuse_raw(graph, scores, **kwargs)

        elif method == Z:
            return diffuse_raw(graph, scores, z=True, **kwargs)

        elif method == ML:
            for score, i, j in scores.__iter__(get_labels=False, get_indices=True):
                if score not in [-1, 0, 1]:
                    raise ValueError("Input scores must be binary.")
                if score == 0:
                    scores.mat[i, j] = -1

            return diffuse_raw(graph, scores, **kwargs)

        elif method == GM:
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
            raise V

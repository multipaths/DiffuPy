# -*- coding: utf-8 -*-

"""This module provides a generalized function as an interface to interact with the different diffusion methods."""

import copy
import inspect
import logging
from typing import Union, Optional, Callable

import networkx as nx
import numpy as np
import pandas as pd

from .kernels import regularised_laplacian_kernel
from .process_input import process_map_and_format_input_data_for_diff
from .process_network import get_kernel_from_network_path, get_graph_from_network_path
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
        network: Union[str, nx.Graph, Matrix],
        method: Union[str, Callable] = Z,
        binarize: Optional[bool] = False,
        threshold: Optional[float] = None,
        absolute_value: Optional[bool] = False,
        p_value: Optional[float] = 0.05,
        kernel_method: Optional[Callable] = regularised_laplacian_kernel
) -> Matrix:
    """Process and format miscellaneous data input and run diffusion over a provided graph network.

    :param input: (miscellaneous format) data input to be processed/formatted.
    :param network: Path to the network or the network Object, as a (NetworkX) graph or as a (diffuPy.Matrix) kernel.
    :param method: Elected method ["raw", "ml", "gm", "ber_s", "ber_p", "mc", "z"] or custom method FUNCTION(network, scores, kargs). By default 'raw'
    :param binarize: If logFC provided in dataset, convert logFC to binary. By default False
    :param threshold: Codify node labels by applying a threshold to logFC in input. By default None
    :param absolute_value: Codify node labels by applying threshold to | logFC | in input. By default False
    :param p_value: Statistical significance. By default 0.05
    :param kernel_method: Callable method for kernel computation.
    """
    log.info(f"{EMOJI} Loading graph. {EMOJI}")

    # Allow custom method function.
    if callable(method):
        diff_call = diffuse_callable(method, network)
        if diff_call:
            return diff_call

    if isinstance(network, nx.Graph):
        kernel = kernel_method(network)
    elif isinstance(network, Matrix):
        kernel = network
    elif isinstance(network, str):
        kernel = get_kernel_from_network_path(network, False, kernel_method=kernel_method)
    else:
        raise IOError(
            f'{EMOJI} The selected network format is not valid neither as a graph or as a kernel. Please ensure you use one of the following formats: '
            f'{GRAPH_FORMATS}'
        )

    log.info(f"{EMOJI} Processing data input. {EMOJI}")

    formated_input_scores = process_map_and_format_input_data_for_diff(input_labels,
                                                                       kernel,
                                                                       method,
                                                                       binarize,
                                                                       absolute_value,
                                                                       p_value,
                                                                       threshold,
                                                                       )

    # Allow custom method function.
    if callable(method):
        return diffuse_callable(method, network, formated_input_scores)

    log.info(f"{EMOJI} Computing the diffusion algorithm. {EMOJI}")

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

        return diffuse_callable(network, network, input_scores, **kwargs)

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
            raise ValueError(f"Method not allowed {method}")

    raise ValueError(f"Method type not allowed {type(method).__name__}")


def diffuse_callable(method_funct, network, scores=None, **kwargs):
    req_args = len(inspect.getfullargspec(method_funct).args) - len(inspect.getfullargspec(method_funct).defaults)
    if isinstance(network, str):
        network = get_graph_from_network_path(network)

    if req_args == 1:
        return method_funct(nx.Graph(network))
    elif req_args == 2 and scores:
        return method_funct(network, scores)
    elif req_args > 2 and scores:
        return method_funct(network, scores, **kwargs)
    elif scores:
        raise ValueError(f"Method function arguments not allowed {type(method_funct).__name__}")
    else:
        return None

# -*- coding: utf-8 -*-


"""Sanity checks for path."""

# .check_metric

import networkx as nx
import numpy as np

from .constants import METHODS
from .matrix import Matrix
from .utils import get_label_list_graph


def _validate_method(method: str) -> None:
    """Ensure that 'method' is a valid character."""
    if not isinstance(method, str):
        raise ValueError(f"The supplied 'method' must be a string. The given argument is a {type(method)}")

    if len(method.split(' ')) > 1:
        raise ValueError(f"Only one method can be supplied, but you supplied {len(method.split(' '))}")

    if method not in METHODS:
        raise ValueError(f"The available methods are {METHODS} but you supplied {method}.")


def _validate_scores(scores: Matrix) -> None:
    """Check scores sanity: Ensures that scores are suitable for diffusion."""
    #  Check labels list
    if not scores.cols_labels:
        raise ValueError("Scores must be a named list but supplied list contains no names.")
    if not scores.rows_labels:
        raise ValueError("Scores must be a named list but supplied list contains no names.")

    #  Check numpy array values type
    if not 'float' and 'int' in str(scores.mat.dtype):
        raise ValueError("The scores in background are not numeric.")

    #  Check each matrix element
    for score, row_label, col_label in iter(scores):
        #  Validate scores

        if score is None:
            raise ValueError("Scores path cannot contain None.")
        elif score is ['NA', 'Nan', 'nan']:
            raise ValueError("Scores path cannot contain NA values.")

        elif isinstance(score, np.int32) or isinstance(score, np.int64):
            score = int(score)
            scores.set_cell_from_labels(row_label, col_label, score)
        elif isinstance(score, np.float32) or isinstance(score, np.float64):
            score = float(score)
            scores.set_cell_from_labels(row_label, col_label, score)

        elif not isinstance(score, float) and not isinstance(score, int):
            raise ValueError("The scores in background are not numeric.")

        #  Validate labels
        if col_label in ['Nan', None]:
            raise ValueError("The scores in background must have row names according to the scored nodes.")
        if row_label in ['Nan', None]:
            raise ValueError("The scores in background must have col names to differentiate score sets.")

    std_mat = Matrix(np.std(scores.mat, axis=0), ['sd'], scores.cols_labels)

    for sd, row_label, col_label in iter(std_mat):
        if sd in ['Nan', None]:
            raise ValueError("Standard deviation in background is NA in column: " + str(col_label))

        # if sd == 0:
        #    raise ValueError("Standard deviation in background is 0 in column:" + str(col_label))


def _validate_graph(graph: nx.Graph) -> None:
    """Check graph sanity: Ensures that 'graph' is a valid NetworkX Graph object."""
    if graph in [None, 'NA', 'Nan']:
        raise ValueError("'graph' missing")

    if not isinstance(graph, nx.Graph):
        raise ValueError(f"The graph must be an NetworkX graph object. The current graph is a {type(graph)}")

    nodes_names = get_label_list_graph(graph, 'name')
    if nodes_names in [None, 'NA', 'Nan']:
        raise ValueError("'graph' must have node names.")

    if any(nodes_names) is None:
        raise ValueError("'graph' cannot have NA as node names")

    if len(np.unique(nodes_names)) != len(nodes_names):
        raise ValueError("'graph' has non-unique names! Please check that the names are unique.")

    if nx.is_directed(graph):
        raise Warning("graph' should be an undirected NetworkX graph object.")

    edge_weights = nx.get_edge_attributes(graph, 'weight')
    if edge_weights:
        if any(edge_weights) is None:
            raise ValueError("'graph' cannot contain NA edge weights, all must have weights.")
        if any(edge_weights) < 0:
            raise Warning("'graph' should not contain negative edge weights.")


def _validate_k(k: Matrix) -> None:
    """Check kernel sanity: Ensures that 'k' is a formally valid kernel."""
    if not isinstance(k, Matrix):
        raise ValueError("'k' must be a Matrix object")

    # Check numeric type.
    if not 'float' and 'int' in str(k.mat.dtype):
        raise ValueError("'k' must be a numeric matrix, but it is not numeric.")

    n_rows = k.mat.shape[0]
    n_cols = k.mat.shape[1]
    if n_rows != n_cols:
        raise ValueError(
            f"'k' must be a square matrix, but it has {str(n_rows)} rows and {str(n_cols)} columns."
        )

    if not k.cols_labels:
        raise ValueError("'k' kernel must have row names.")

    if not k.rows_labels:
        raise ValueError("'k' kernel must have column names.")

    if k.rows_labels != k.cols_labels:
        raise ValueError("'k' rownames and colnames must be identical.")

    # TODO: Bottleneck
    # for score, row_label, col_label in iter(k):
    #     # print(k)
    #     # print(np.int64)
    #     # print(type(score))
    #     # print(type(score) == np.int64)
    #     # print(isinstance(score, np.int64))
    #     # print(np.issubdtype(score, np.dtype('int64')))
    #     # print(np.issubdtype(score, int))
    #
    #     # if isinstance(score, np.int32) or isinstance(score, np.int64):
    #     # if type(score) == np.int32 or type(score) == np.int64:
    #
    #     if np.issubdtype(score, int):
    #         score = int(score)
    #         k.set_from_labels(row_label, col_label, score)
    #
    #     elif np.issubdtype(score, float):
    #         score = float(score)
    #         k.set_from_labels(row_label, col_label, score)
    #
    #     elif not isinstance(score, float) and not isinstance(score, int):
    #         raise ValueError("'k' must be a numeric matrix, but it is not numeric.")
    #
    #     if score in ['Nan', None]:
    #         raise ValueError("Scores path cannot contain NA, but background does.")
    #
    #     if col_label in ['Nan', None] or row_label in ['Nan', None]:
    #         raise ValueError("'k' dimnames cannot be NA.")

    if len(np.unique(k.rows_labels)) != len(k.rows_labels):
        raise ValueError("'k' cannot contain duplicated row names.")

    if len(np.unique(k.cols_labels)) != len(k.cols_labels):
        raise ValueError("'k' cannot contain duplicated column names.")

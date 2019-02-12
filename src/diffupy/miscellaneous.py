# -*- coding: utf-8 -*-

"""Miscellaneous utils of the package."""

from typing import List

import networkx as nx
import numpy as np


def get_laplacian(graph: nx.Graph, normalized: bool = False) -> np.ndarray:
    """Return Laplacian matrix."""
    if nx.is_directed(graph):
        raise ValueError('Graph must be undirected')

    # Normalize matrix
    if normalized:
        return nx.normalized_laplacian_matrix(graph).toarray()

    return nx.laplacian_matrix(graph).toarray()


def set_diagonal_matrix(matrix, d):
    """"""
    for j, row in enumerate(matrix):
        for i, x in enumerate(row):
            if i == j:
                matrix[j][i] = d[i]
            else:
                matrix[j][i] = x
    return matrix


def get_label_list_graph(graph: nx.Graph, label: str) -> List:
    """Return graph labels."""
    return [
        value
        for value in nx.get_node_attributes(graph, label).values()
    ]


def get_label_ix_mapping(labels):
    """Get label to mat index mappings."""
    return {
        # label.decode('utf-8'): i
        str(label).replace('b', '').replace("'", '').replace('"', ''): i
        for i, label in enumerate(labels)
    }

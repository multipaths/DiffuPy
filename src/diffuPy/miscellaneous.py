# -*- coding: utf-8 -*-

"""Miscellaneous utils of the package."""

from typing import List

import networkx as nx
import numpy as np

import warnings

import logging
log = logging.getLogger(__name__)

def get_laplacian(graph: nx.Graph, normalized: bool = False) -> np.ndarray:
    """Return Laplacian matrix."""
    if nx.is_directed(graph):
        graph = graph.to_undirected()
        warnings.warn('Graph must be undirected, so it is converted to undirected.')

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

    if isinstance(list(graph.nodes(data=True))[0], tuple):
        labels = []
        for node, _ in graph.nodes(data=True):
            if hasattr(node, 'name') and node.name is not None:
                if node.name.lower() == "":
                    log.warning('Empty attribute name.' + str(node))
                    labels.append(str(node))

                else:
                    labels.append(node.name.lower())

            elif hasattr(node, 'id') and node.id is not None:
                if node.id.lower() == "":
                    log.warning('Empty attribute id.' + str(node))
                    labels.append(str(node))

                else:
                    labels.append(node.id.lower())
                    log.warning('Node labeled with id.' + node.id.lower())

            else:
                if str(node) == "":
                    log.warning('Node with no info.')
                else:
                    labels.append(str(node))
                    log.warning('Node name nor id not labeled. ' + str(node))

        return labels

    elif nx.get_node_attributes(graph, label).values():
        return [
            value
            for value in nx.get_node_attributes(graph, label).values()
        ]

    else:
        raise Warning('Could not get a label list from graph.')



def get_label_ix_mapping(labels):
    """Get label to mat index mappings."""
    mapping = {}
    labels_decode = []

    for i, label in enumerate(labels):
        if not isinstance(label, str):
            label = label.decode('utf-8').replace('"', '')

        mapping[label] = i
        labels_decode.append(label)

    return mapping, labels_decode

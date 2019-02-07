# -*- coding: utf-8 -*-

"""Miscellaneous utils of the package."""

import networkx as nx

from .kernel import get_laplacian


def get_label_list_graph(graph):
    """Return graph labels."""
    return [
        v
        for k, v in nx.get_node_attributes(graph, 'name')
    ]


def get_label_id_mapping(labels_row, labels_col=None):
    """Get label id mappings."""
    if not labels_col:
        return {
            label: i
            for i, label in enumerate(labels_row)
        }

    return {
        label_row: {
            label_col: (j, i)
            for j, label_col in enumerate(labels_col)}
        for i, label_row in enumerate(labels_row)
    }


# TODO: borrar de aqui
class LaplacianMatrix(Matrix):
    def __init__(self, graph, normalized=False, name=''):
        l_mat = get_laplacian(graph, normalized)
        Matrix.__init__(self, l_mat, name=name, dupl=True, graph=graph)

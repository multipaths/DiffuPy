# -*- coding: utf-8 -*-

"""Miscellaneous utils of the package."""
import itertools
import logging
import warnings
from typing import List

import networkx as nx
import numpy as np

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
                    log.warning(f'Empty attribute name: {node.as_bel()}')
                    labels.append(node.as_bel())

                else:
                    labels.append(node.name.lower())

            elif hasattr(node, 'id') and node.id is not None:
                if node.id.lower() == "":
                    log.warning(f'Empty attribute id: {node.as_bel()}')
                    labels.append(node.as_bel())

                else:
                    labels.append(node.id.lower())
                    log.warning('Node labeled with id.' + node.id.lower())

            else:
                if node.as_bel() == "":
                    log.warning('Node with no info.')
                else:
                    labels.append(node.as_bel())
                    log.warning(f'Node name nor id not labeled: {node.as_bel()}')

        return labels

    elif nx.get_node_attributes(graph, label).values():
        return [
            value
            for value in nx.get_node_attributes(graph, label).values()
        ]

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


def print_dict_dimensions(entities_db, title):
    """Print dimension of the dictionary"""
    total = 0
    print(title)
    for k1, v1 in entities_db.items():
        m = ''
        if isinstance(v1, dict):
            for k2, v2 in v1.items():
                m += f'{k2}({len(v2)}), '
                total += len(v2)
        else:
            m += f'{len(v1)} '
            total += len(v1)

        print(f'Total number of {k1}: {m} ')

    print(f'Total: {total} ')


def get_labels_set_from_dict(entities):
    if isinstance(list(entities.values())[0], dict):
        # TODO: Check
        return set(itertools.chain.from_iterable(itertools.chain.from_iterable(entities.values())))
    else:
        return set(itertools.chain.from_iterable(entities.values()))


def reduce_dict_dimension(dict):
    """Reduce dictionary dimension."""
    return {
        k: set(itertools.chain.from_iterable(entities.values()))
        for k, entities in dict.items()
    }


def check_substrings(dataset_nodes, db_nodes):
    mapping_substrings = set()

    for entity in dataset_nodes:
        if isinstance(entity, tuple):
            for subentity in entity:
                for entity_db in db_nodes:
                    if isinstance(entity_db, tuple):
                        for subentity_db in entity_db:
                            if subentity_db in subentity or subentity in subentity_db:
                                mapping_substrings.add(entity_db)
                                break
                        break
                    else:
                        if entity_db in subentity or subentity in entity_db:
                            mapping_substrings.add(entity_db)
                            break
        else:
            for entity_db in db_nodes:
                if isinstance(entity_db, tuple):
                    for subentity_db in entity_db:
                        if subentity_db in entity or entity in subentity_db:
                            mapping_substrings.add(entity_db)
                            break
                    break
                else:
                    if entity_db in entity or entity in entity_db:
                        mapping_substrings.add(entity_db)
                        break

    return mapping_substrings

# -*- coding: utf-8 -*-

"""Miscellaneous utils of the package."""

import random

import itertools
import logging
import warnings
from typing import List

import networkx as nx
import numpy as np
import pybel


log = logging.getLogger(__name__)


def get_laplacian(graph: nx.Graph, normalized: bool = False) -> np.ndarray:
    """Return Laplacian matrix."""
    if nx.is_directed(graph):
        warnings.warn('Since graph is directed, it will be converted to an undirected graph.')
        graph = graph.to_undirected()

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

def get_label_node(node: nx.Graph.node) -> str:
    if hasattr(node, 'name') and node.name is not None:
        if node.name.lower() == "":
            log.debug(f'Empty attribute name: {node.as_bel()}')
            return node.as_bel()

        else:
            return node.name.lower()

    elif hasattr(node, 'id') and node.id is not None:
        if node.id.lower() == "":
            log.debug(f'Empty attribute id: {node.as_bel()}')
            return node.as_bel()

        else:
            log.debug('Node labeled with id.' + node.id.lower())
            return node.id.lower()

    else:
        if node.as_bel() == "":
            log.debug('Node with no info.')
        else:
            log.debug(f'Node name nor id not labeled: {node.as_bel()}')
            return node.as_bel()


def get_label_list_graph(graph: nx.Graph, label: str) -> List:
    """Return graph labels."""
    if isinstance(graph, pybel.BELGraph):
        labels = []
        for node, _ in graph.nodes(data=True):
            labels.append(get_label_node(node))

        return labels

    elif nx.get_node_attributes(graph, label).values():
        return [
            value
            for value in nx.get_node_attributes(graph, label).values()
        ]

    raise Warning('Could not get a label list from graph.')


def get_label_ix_mapping(labels):
    """Get label to mat index mappings."""
    return {label:i for i, label in enumerate(labels)}

def decode_labels(labels):
    """Validate labels."""
    labels_decode = []

    for label in labels:
        if not isinstance(label, str):
            label = label.decode('utf-8').replace('"', '')

        labels_decode.append(label)

    return labels_decode

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

def get_simplegraph_from_multigraph(multigraph):

    G = nx.Graph()
    for u,v,data in multigraph.edges(data=True):
        u = get_label_node(u)
        v = get_label_node(v)

        w = data['weight'] if 'weight' in data else 1.0
        if G.has_edge(u,v):
            G[u][v]['weight'] += w
        else:
            G.add_edge(u, v, weight=w)

    return G

def split_random_two_subsets(to_split):
    half_1 = random.sample(population=list(to_split), k=int(len(to_split) / 2))
    half_2 = list(set(to_split) - set(half_1))

    return half_1, half_2


def split_random_three_subsets(to_split):
    half_1 = random.sample(population=list(to_split), k=int(len(to_split) / 3))
    half_2, half_3 = split_random_two_subsets(list(set(to_split) - set(half_1)))

    return half_1, half_2, half_3


def get_three_venn_intersections(set1, set2, set3):
    set1, set2, set3 = set(set1), set(set2), set(set3)
    set1_set2 = set1.intersection(set2)
    set1_set3 = set1.intersection(set3)
    core = set1_set3.intersection(set1_set2)

    set1_set2 = set1_set2 - core
    set1_set3 = set1_set3 - core
    set2_set3 = set2.intersection(set3) - core

    return {'unique_set1': set1 - set1_set2 - set1_set3 - core,
            'unique_set2': set2 - set1_set2 - set2_set3 - core,
            'set1_set2': set1_set2,
            'unique_set3': set3 - set1_set3 - set2_set3 - core,
            'set1_set3': set1_set3,
            'set2_set3': set2_set3,
            'core': core,
            }


def random_disjoint_intersection_two_subsets(unique_set1, unique_set2, intersection):
    set1, set2 = split_random_two_subsets(intersection)

    return unique_set1|set(set1), unique_set2|set(set2)


def random_disjoint_intersection_three_subset(set1, set2, set3):
    intersections = get_three_venn_intersections(set1, set2, set3)

    set1, set2 = random_disjoint_intersection_two_subsets(intersections['unique_set1'],
                                                          intersections['unique_set2'],
                                                          intersections['set1_set2']
                                                          )


    set1, set3 = random_disjoint_intersection_two_subsets(set1,
                                                          intersections['unique_set3'],
                                                          intersections['set1_set3']
                                                          )

    set2, set3 = random_disjoint_intersection_two_subsets(set2,
                                                          set3,
                                                          intersections['set2_set3']
                                                          )

    set1_core, set2_core, set3_core = split_random_three_subsets(intersections['core'])

    print(len(set1_core))
    print(len(set2_core))
    print(len(set3_core))


    return set1|set(set1_core), set2|set(set2_core), set3|set(set3_core)

# -*- coding: utf-8 -*-

"""Miscellaneous utils of the package."""

import json
import logging
import pickle
import random
import warnings
from typing import List

import networkx as nx
import numpy as np
import pandas as pd
import pybel
from networkx import Graph

from .constants import *
from .constants import CSV, TSV, GRAPH_FORMATS

log = logging.getLogger(__name__)


def from_dataframe_file(path: str, fmt: str) -> pd.DataFrame:
    """Read network file."""
    format_checker(fmt)

    return pd.read_csv(
        path,
        header=0,
        sep=FORMAT_SEPARATOR_MAPPING[CSV] if fmt == CSV else FORMAT_SEPARATOR_MAPPING[TSV]
    )


def from_json(path: str):
    """Read from json file."""
    with open(path) as f:
        return json.load(f)


def from_pickle(input_path):
    """Read from pickle file."""
    with open(input_path, 'rb') as f:
        unpickler = pickle.Unpickler(f)
        return unpickler.load()


def from_nparray_to_df(nparray: np.ndarray) -> pd.DataFrame:
    """Convert numpy array to data frame."""
    return pd.DataFrame(data=nparray[1:, 1:],
                        index=nparray[1:, 0],
                        columns=nparray[0, 1:])


def get_laplacian(graph: Graph, normalized: bool = False) -> np.ndarray:
    """Return Laplacian matrix."""
    if nx.is_directed(graph):
        warnings.warn('Since graph is directed, it will be converted to an undirected graph.')
        graph = graph.to_undirected()

    # Normalize matrix
    if normalized:
        return nx.normalized_laplacian_matrix(graph).toarray()

    return nx.laplacian_matrix(graph).toarray()


def set_diagonal_matrix(matrix: np.ndarray, d: list) -> np.ndarray:
    """Set diagonal matrix."""
    for j, row in enumerate(matrix):
        for i, x in enumerate(row):
            if i == j:
                matrix[j][i] = d[i]
            else:
                matrix[j][i] = x
    return matrix


def get_label_node(node: pybel.dsl.BaseAbundance) -> str:
    """Get label node."""
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

    elif graph.nodes:
        return graph.nodes()

    raise Warning('Could not get a label list from graph.')


def get_repeated_labels(labels):
    """Get duplicate labels."""
    seen = set()
    rep = []
    for x in labels:
        if x in seen:
            rep.append(x)
        seen.add(x)

    return rep


def get_label_ix_mapping(labels):
    """Get label to mat index mappings."""
    return {label: i for i, label in enumerate(labels)}


def get_label_scores_mapping(labels, scores):
    """Get label to scores mapping."""
    return {label: scores[i] for i, label in enumerate(labels)}


def get_idx_scores_mapping(scores):
    """Get mat index to scores mapping."""
    return {i: score for i, score in enumerate(scores)}


def decode_labels(labels):
    """Validate labels."""
    labels_decode = []

    for label in labels:
        if not isinstance(label, str):

            if isinstance(label, int):
                label = str(label)
            else:
                label = label.decode('utf-8').replace('"', '')

        labels_decode.append(label)

    return labels_decode


def print_dict_dimensions(entities_db, title):
    """Print dimension of the dictionary."""
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


def format_checker(fmt: str, fmt_list: list = GRAPH_FORMATS) -> None:
    """Check formats."""
    if fmt not in fmt_list:
        raise ValueError(
            f'The selected sep {fmt} is not valid. Please ensure you use one of the following formats: '
            f'{fmt_list}'
        )


def get_random_key_from_dict(d):
    return random.choice(list(d.keys()))


def get_random_value_from_dict(d):
    return d[get_random_key_from_dict(d)]

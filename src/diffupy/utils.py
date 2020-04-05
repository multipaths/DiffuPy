# -*- coding: utf-8 -*-

"""Miscellaneous utils of the package."""

import json
import logging
import pickle
import warnings
from typing import List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import pybel

from networkx import DiGraph, read_graphml, read_gml, node_link_graph, read_edgelist

from .constants import *
from .constants import CSV, TSV, GRAPHML, GML, BEL, BEL_PICKLE, NODE_LINK_JSON, EMOJI, FORMATS


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


def get_simple_graph_from_multigraph(multigraph):
    """Convert undirected graph from multigraph."""
    graph = nx.Graph()
    for u, v, data in multigraph.edges(data=True):
        u = get_label_node(u)
        v = get_label_node(v)

        w = data['weight'] if 'weight' in data else 1.0
        if graph.has_edge(u, v):
            graph[u][v]['weight'] += w
        else:
            graph.add_edge(u, v, weight=w)

    return graph


"""Check formats of networks """


def _format_checker(fmt: str) -> None:
    """Check column sep."""
    if fmt not in FORMATS:
        raise ValueError(
            f'The selected sep {fmt} is not valid. Please ensure you use one of the following formats: '
            f'{FORMATS}'
        )


"""Process networks"""


def _read_network_file(path: str, fmt: str) -> pd.DataFrame:
    """Read network file."""
    _format_checker(fmt)

    df = pd.read_csv(
        path,
        header=0,
        sep=FORMAT_SEPARATOR_MAPPING[CSV] if fmt == CSV else FORMAT_SEPARATOR_MAPPING[TSV]
    )

    if SOURCE not in df.columns or TARGET not in df.columns:
        raise ValueError(
            f'Ensure that your file contains columns for {SOURCE} and {TARGET}. The column for {RELATION} is optional'
            f'and can be omitted.'
        )

    return df


def process_network(path: str, sep: str) -> DiGraph:
    """Return network from dataFrame."""
    _format_checker(sep)

    df = _read_network_file(path, sep)

    graph = DiGraph()

    for index, row in df.iterrows():

        # Get node names from data frame
        sub_name = row[SOURCE]
        obj_name = row[TARGET]

        if RELATION in df.columns:

            relation = row[RELATION]

            # Store edge in the graph
            graph.add_edge(
                sub_name, obj_name,
                relation=relation,
            )

        else:
            graph.add_edge(
                sub_name, obj_name,
            )

    return graph


def load_json_file(path: str) -> DiGraph:
    """Read json file."""
    with open(path) as f:
        return json.load(f)


def from_pickle(input_path):
    """Read from pickle file."""
    with open(input_path, 'rb') as f:
        unpickler = pickle.Unpickler(f)
        return unpickler.load()


def process_network_from_cli(path: str) -> nx.Graph:
    """Load network from path."""
    if path.endswith(CSV):
        graph = process_network(path, CSV)

    elif path.endswith(TSV):
        graph = process_network(path, TSV)

    elif path.endswith(GRAPHML):
        graph = read_graphml(path)

    elif path.endswith(GML):
        graph = read_gml(path)

    elif path.endswith(BEL):
        graph = pybel.from_path(path)

    elif path.endswith(BEL_PICKLE):
        graph = pybel.from_pickle(path)

    elif path.endswith(EDGE_LIST):
        graph = read_edgelist(path)

    elif path.endswith(NODE_LINK_JSON):
        data = load_json_file(path)
        graph = node_link_graph(data)

    else:
        raise IOError(
            f'{EMOJI} The selected format is not valid. Please ensure you use one of the following formats: '
            f'{FORMATS}'
        )
    return graph


def process_kernel_from_cli(path: str):
    """Process kernel from cli."""
    # TODO process different kinds of input format kernel
    return from_pickle(path)

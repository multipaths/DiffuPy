# -*- coding: utf-8 -*-

"""Miscellaneous utils of the package."""

import json
import logging
import warnings
from typing import List, Optional

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


def _format_checker(format: str) -> None:
    """Check column sep."""
    if format not in FORMATS:
        raise ValueError(
            f'The selected sep {format} is not valid. Please ensure you use one of the following formats: '
            f'{FORMATS}'
        )


def _read_network_file(path: str, format: str) -> pd.DataFrame:
    """Read network file."""
    _format_checker(format)

    df = pd.read_csv(
        path,
        header=0,
        sep=FORMAT_SEPARATOR_MAPPING[CSV] if format == CSV else FORMAT_SEPARATOR_MAPPING[TSV]
    )

    if SOURCE and TARGET not in df.columns:
        raise ValueError(
            f'Ensure that your file contains columns for {SOURCE} and {TARGET}. The column for {RELATION} is optional'
            f'and can be omitted.'
        )

    return df


def process_network(path: str, sep: str) -> DiGraph:
    """Return network from dataframe."""
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
                relation=relation
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


def process_network_from_cli(network: str) -> nx.Graph:
    """Load network from path."""
    if network.endswith(CSV):
        graph = process_network(network, CSV)

    elif network.endswith(TSV):
        graph = process_network(network, TSV)

    elif network.endswith(GRAPHML):
        graph = read_graphml(network)

    elif network.endswith(GML):
        graph = read_gml(network)

    elif network.endswith(BEL):
        graph = pybel.from_path(network)

    elif network.endswith(BEL_PICKLE):
        graph = pybel.from_pickle(network)

    elif network.endswith(EDGE_LIST):
        graph = read_edgelist(network)

    elif network.endswith(NODE_LINK_JSON):
        data = load_json_file(network)
        graph = node_link_graph(data)

    else:
        raise IOError(
            f'{EMOJI} The selected format {format} is not valid. Please ensure you use one of the following formats: '
            f'{FORMATS}'
        )
    return graph


def _process_input(path: str, format: str) -> pd.DataFrame:
    """Read input file and ensure necessary columns exist."""
    _format_checker(format)

    df = pd.read_csv(
        path,
        header=0,
        sep=FORMAT_SEPARATOR_MAPPING[CSV] if format == CSV else FORMAT_SEPARATOR_MAPPING[TSV]
    )
    if not (df.columns.isin([NODE, EXPRESSION, P_VALUE]).all()):
        raise ValueError(
            f'Ensure that your file contains columns for {NODE}, {EXPRESSION} and {P_VALUE}.'
        )

    return df


def prepare_input_data(df: pd.DataFrame, method: str, absolute_value=False, p_value=0.05, threshold=None) -> \
        pd.DataFrame:
    """Prepare input data for diffusion."""

    # Prepare input data dataFrame for quantitative diffusion methods
    if method == RAW or method == Z:
        return _prepare_quantitative_input_data(df, absolute_value, p_value, threshold)

    # Prepare input data dataFrame for non-quantitative diffusion methods
    elif method == ML or method == GM:
        return _prepare_non_quantitative_input_data(df, p_value, threshold)

    else:
        # TODO: ber_s, ber_p, mc?
        raise NotImplementedError('This diffusion method has not yet been implemented.')


def _prepare_quantitative_input_data(df: pd.DataFrame, absolute_value: bool, p_value: int, threshold: Optional[int]) \
        -> pd.DataFrame:
    """Prepare input data for quantitative diffusion methods."""
    # Threshold value is provided
    if threshold:

        # Label nodes with |expression values| over threshold as 1 and below threshold as 0
        if absolute_value is True:
            return _filter_by_abs_val(df, threshold)

        # Label nodes with values over threshold as 1, below threshold as 0 and remove nodes with negative values
        return _filter_by_threshold(df, threshold)

    # If no threshold is provided, pass statistically significant expression values as labels
    # Create a copy of column 'Expression' called 'Label' and add column 'Label' to DataFrame
    df[LABEL] = df[EXPRESSION]

    # Create dataFrame that has statistically significant expression values as labels
    significant_values_df = df.loc[df[P_VALUE] <= p_value].copy()

    return significant_values_df[[NODE, LABEL]]


def _prepare_non_quantitative_input_data(df: pd.DataFrame, p_value: int, threshold: Optional[int]) -> pd.DataFrame:
    """Process input data for non-quantitative diffusion methods."""

    # TODO: is p-value mandatory or optional?
    if p_value and threshold:

        # Label nodes whose expression passes the threshold with 1
        df.loc[(df[EXPRESSION] >= threshold), LABEL] = 1

        # TODO: option to label those that fall below threshold as 0?
        # Label nodes whose expression falls below the threshold with -1
        df.loc[(df[EXPRESSION] < threshold), LABEL] = -1

        # Label nodes whose expression is not statistically significant with 0
        df.loc[(df[P_VALUE] > p_value), LABEL] = 0

        return df[[NODE, LABEL]]

    # TODO: is p-value mandatory or optional?
    elif p_value:

        # Label nodes whose expression is statistically significant with 1
        df.loc[(df[P_VALUE] <= p_value), LABEL] = 1

        # Label nodes whose expression is not statistically significant with 0
        df.loc[(df[P_VALUE] > p_value), LABEL] = 0

        return df[[NODE, LABEL]]

    # TODO: do we need 2 options here: 1) if below threshold 0 and 2) if below threshold -1? what is this based on?
    # TODO: what does user specify to get these 2 options? is one for ml and the other gm?
    elif threshold:

        # Label nodes whose expression passes the threshold with 1
        df.loc[(df[EXPRESSION] >= threshold), LABEL] = 1

        # Label nodes whose expression falls below the threshold with -1
        df.loc[(df[EXPRESSION] < threshold), LABEL] = -1

        return df[[NODE, LABEL]]


def _filter_by_abs_val(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """Label nodes as 1 or 0 if |expression values| fall above or below the value of a threshold, respectively."""
    # Get absolute values of all expression values
    df[ABSOLUTE_VALUE_EXP] = df[EXPRESSION].abs()

    # Label nodes with |expression values| falling above the threshold with 1
    df.loc[(df[ABSOLUTE_VALUE_EXP] >= threshold), LABEL] = 1

    # Label nodes with |expression values| falling below the threshold with 0
    df.loc[(df[ABSOLUTE_VALUE_EXP] < threshold), LABEL] = 0

    # TODO: remove nodes that are not significant?
    # Get nodes and labels dataFrame
    return df[[NODE, LABEL]]


def _filter_by_threshold(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """Filter expression values in dataset by a threshold and set node labels."""
    # Label nodes with expression values falling above the threshold with 1
    df.loc[(df[EXPRESSION] >= threshold), LABEL] = 1

    # Label nodes with expression values falling below the threshold as 0
    df.loc[(df[EXPRESSION] < threshold), LABEL] = 0

    # Only keep nodes with non-negative expression values
    df = df.loc[df[EXPRESSION] > 0]

    # TODO: remove nodes that are not significant?
    # Get nodes and labels dataFrame
    return df[[NODE, LABEL]]

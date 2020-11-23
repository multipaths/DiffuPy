# -*- coding: utf-8 -*-

"""Miscellaneous utils of the package."""
import inspect
import logging
from typing import Tuple, Union, List, Optional, Callable

import numpy as np
import pandas as pd
import pybel
from diffupy.matrix import Matrix, MatrixFromDataFrame, MatrixFromDict, MatrixFromNumpyArray
from diffupy.utils import from_dataframe_file, format_checker, from_pickle, get_label_node, from_json
from networkx import DiGraph, Graph, read_graphml, read_gml, node_link_graph, read_edgelist, nx
from pybel.struct.mutation.induction.annotations import get_subgraph_by_annotation_value

from .constants import *
from .constants import CSV, TSV, GRAPHML, GML, BEL, PICKLE, EMOJI, GRAPH_FORMATS
from .kernels import regularised_laplacian_kernel

log = logging.getLogger(__name__)

"""Process network as undefined format (could represented as a graph or as a kernel)"""


def get_kernel_and_graph_from_network_path(path: str,
                                           filter_network_database: Optional[List[str]] = None,
                                           filter_network_omic: Optional[List[str]] = None,
                                           kernel_method: Optional[Callable] = regularised_laplacian_kernel
                                           ) -> Tuple[Matrix, Graph]:
    """Load network provided in cli as a kernel and as a graph."""
    graph = None
    kernel = None

    if path.endswith(KERNEL_FORMATS):
        try:
            graph = process_graph_from_file(path)

        except TypeError:
            kernel = process_kernel_from_file(path)

    elif path.endswith(GRAPH_FORMATS):
        graph = process_graph_from_file(path)

        if filter_network_database:
            graph = get_subgraph_by_annotation_value(graph,
                                                     'database',
                                                     filter_network_database
                                                     )

        if filter_network_omic:
            graph = get_subgraph_by_annotation_value(graph,
                                                     'enity_type',
                                                     filter_network_omic
                                                     )

    else:
        raise IOError(
            f'{EMOJI} The selected network format is not valid neither as a graph or as a kernel. Please ensure you use one of the following formats: '
            f'{GRAPH_FORMATS}'
        )

    if kernel is None and graph is not None:
        kernel = kernel_method(graph)

    if kernel is not None and graph is None:
        graph = kernel.to_nx_graph()

    return kernel, graph


def get_kernel_from_network_path(path: str,
                                 normalized: bool = False,
                                 filter_network_database: Optional[List[str]] = None,
                                 filter_network_omic: Optional[List[str]] = None,
                                 kernel_method: Optional[Callable] = regularised_laplacian_kernel,
                                 ) -> Matrix:
    """Load network provided in cli (as a graph or as a kernel) retrieving a kernel."""
    if path.endswith(KERNEL_FORMATS) and filter_network_omic is None and filter_network_database is None:
        try:
            graph = process_graph_from_file(path)
            if filter_network_omic or filter_network_database:
                raise ValueError(
                    "The provided network can not be filtered, since has been provided as a kernel. "
                    "For filtering, please provide the network formated as a graph.")
        except TypeError:
            return process_kernel_from_file(path)

    elif path.endswith(GRAPH_FORMATS):
        graph = process_graph_from_file(path)

        if filter_network_database:
            graph = get_subgraph_by_annotation_value(graph,
                                                     'database',
                                                     filter_network_database
                                                     )

        if filter_network_omic:
            graph = get_subgraph_by_annotation_value(graph,
                                                     'enity_type',
                                                     filter_network_omic
                                                     )

    else:
        raise IOError(
            f'{EMOJI} The selected network format is not valid neither as a graph or as a kernel. Please ensure you use one of the following formats: '
            f'{GRAPH_FORMATS}'
        )

    return get_kernel_from_graph(graph, kernel_method, normalized)


def filter_graph(graph: nx.Graph, filter_network_database=None, filter_network_omic=None) -> nx.Graph:
    """Filter graph by database or/and by omic."""
    if filter_network_database:
        graph = get_subgraph_by_annotation_value(graph,
                                                 'database',
                                                 filter_network_database
                                                 )

    if filter_network_omic:
        graph = get_subgraph_by_annotation_value(graph,
                                                 'enity_type',
                                                 filter_network_omic
                                                 )
    return graph


def get_kernel_from_graph(graph, kernel_method, normalized=False):
    """Get kernel from graph given a kernel method."""
    if 'normalized' in inspect.getfullargspec(kernel_method).args:
        return kernel_method(graph, normalized=normalized)
    else:
        return kernel_method(graph)


def get_graph_from_network_path(path: str) -> Graph:
    """Load network provided in cli as a graph."""
    if path.endswith(KERNEL_FORMATS):
        try:
            return process_graph_from_file(path)

        except TypeError:
            kernel = process_kernel_from_file(path)

    elif path.endswith(GRAPH_FORMATS):
        return process_graph_from_file(path)

    else:
        raise IOError(
            f'{EMOJI} The selected network format is not valid neither as a graph or as a kernel. Please ensure you use one of the following formats: '
            f'{GRAPH_FORMATS}'
        )

    return kernel.to_nx_graph()


"""Process input file formats"""


def process_graph_from_file(path: str) -> Graph:
    """Load network from path."""
    if path.endswith(CSV) or path.endswith(TSV):
        graph = get_graph_from_df(path, CSV)

    elif path.endswith(TSV):
        graph = get_graph_from_df(path, TSV)

    elif path.endswith(PICKLE):
        graph = pybel.from_pickle(path)

    elif path.endswith(GRAPHML):
        graph = read_graphml(path)

    elif path.endswith(GML):
        graph = read_gml(path, label='id')

    elif path.endswith(BEL):
        graph = pybel.from_path(path)

    elif path.endswith(EDGE_LIST):
        graph = read_edgelist(path)

    elif path.endswith(JSON):
        data = from_json(path)
        graph = node_link_graph(data)
    else:
        raise IOError(
            f'{EMOJI} The selected graph format is not valid. Please ensure you use one of the following formats: '
            f'{GRAPH_FORMATS}'
        )

    n_isolates = nx.number_of_isolates(graph)
    if n_isolates != 0:
        log.info(
            f'{EMOJI} Graph loaded with {n_isolates} isolate nodes, which will be removed since are not relevant'
            f'for the diffusion procedures. {EMOJI} \n'
        )
        graph.remove_nodes_from({
            node
            for node in nx.isolates(graph)
        })

    log.info(
        f'{EMOJI} Graph loaded with: \n'
        f'{graph.number_of_nodes()} nodes\n'
        f'{graph.number_of_edges()} edges\n'
        f'{EMOJI}'
    )

    return graph


def process_kernel_from_file(path: str) -> Matrix:
    """Load kernel from path."""
    if path.endswith(CSV):
        raw_kernel = MatrixFromDataFrame(from_dataframe_file(path, CSV))

    elif path.endswith(TSV):
        raw_kernel = MatrixFromDataFrame(from_dataframe_file(path, TSV))

    elif path.endswith(PICKLE):
        raw_kernel = from_pickle(path)

    elif path.endswith(JSON):
        raw_kernel = from_json(path)

    else:
        raise IOError(
            f'{EMOJI} The selected kernel format is not valid. Please ensure you use one of the following formats: '
            f'{KERNEL_FORMATS}'
        )

    # Check imported type of kernel
    if isinstance(raw_kernel, Matrix):
        kernel = raw_kernel

    elif isinstance(raw_kernel, dict):
        kernel = MatrixFromDict(raw_kernel)

    elif isinstance(raw_kernel, pd.DataFrame):
        kernel = MatrixFromDataFrame(raw_kernel)

    elif isinstance(raw_kernel, np.ndarray):
        kernel = MatrixFromNumpyArray(raw_kernel)

    else:
        raise IOError(
            f'{EMOJI} The imported kernel type is not valid. Please ensure it is provided as a diffupy '
            f'Matrix, a Dict, NumpyArray or Pandas DataFrame. '
        )

    log.info(
        f'{EMOJI} Kernel loaded with: \n'
        f'{len(kernel.rows_labels)} nodes\n'
        f'{EMOJI}'
    )

    return kernel


def get_simple_graph_from_multigraph(multigraph: Union[nx.MultiGraph, nx.MultiDiGraph]) -> nx.Graph:
    """Convert undirected graph from multigraph."""
    graph = Graph()
    for u, v, data in multigraph.edges(data=True):
        u = get_label_node(u)
        v = get_label_node(v)

        w = data['weight'] if 'weight' in data else 1.0
        if graph.has_edge(u, v):
            graph[u][v]['weight'] += w
        else:
            graph.add_edge(u, v, weight=w)

    return graph


def get_graph_from_df(path: str, sep: str) -> DiGraph:
    """Return network from dataFrame."""
    format_checker(sep)

    df = from_dataframe_file(path, sep)

    if SOURCE not in df.columns or TARGET not in df.columns:
        raise ValueError(
            f'Ensure that your file contains columns for {SOURCE} and {TARGET}. The column for {RELATION} is optional'
            f'and can be omitted.'
        )

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

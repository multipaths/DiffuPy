# -*- coding: utf-8 -*-

"""Command line interface for diffuPy."""

import logging
import os
import pickle
import time

import click
import pybel
from networkx import read_graphml, read_gml, node_link_graph

from diffupy.constants import (
    CSV, TSV, FORMATS, GRAPHML, GML, BEL, BEL_PICKLE, NODE_LINK_JSON
)
from diffupy.utils import process_network, load_json_file
from .constants import OUTPUT, METHODS, EMOJI
from .kernels import regularised_laplacian_kernel

logger = logging.getLogger(__name__)


@click.group(help='DiffuPy')
def main():
    """Command line interface for diffuPy."""
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")


@main.command()
@click.option(
    '-n', '--network',
    help='Input network',
    required=True,
    type=click.Path(exists=True, dir_okay=False)
)
@click.option(
    '-o', '--output',
    help='Output path to store the generated kernel pickle',
    default=OUTPUT,
    show_default=True,
    type=click.Path(exists=True, file_okay=False)
)
@click.option('-l', '--log', is_flag=True, help='Activate debug mode')
def kernel(
        network: str,
        output: str = OUTPUT,
        log: bool = None
):
    """Generate a kernel for a given network.

    :param path: Path to retrieve the input/source network
    :param output: Path to store the output/generated kernel pickle
    :param isolates: Include isolates
    :param log: Activate debug mode
    """
    # Configure logging level
    if log:
        logging.basicConfig(level=logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
        logger.setLevel(logging.INFO)

    click.secho(f'{EMOJI} Loading graph from {network} {EMOJI}')

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

    elif network.endswith(NODE_LINK_JSON):
        data = load_json_file(network)
        graph = node_link_graph(data)

    else:
        raise IOError(
            f'{EMOJI} The selected format {format} is not valid. Please ensure you use one of the following formats: '
            f'{FORMATS}'
        )

    click.secho(f'{EMOJI} Calculating regulatised Laplacian kernel. This might take a while... {EMOJI}')

    exe_t_0 = time.time()
    background_mat = regularised_laplacian_kernel(graph)
    exe_t_f = time.time()

    output_file = os.path.join(output, f'{network.split("/")[-1]}.pickle')

    # Export numpy array
    with open(output_file, 'wb') as file:
        pickle.dump(background_mat, file, protocol=4)

    running_time = exe_t_f - exe_t_0

    click.secho(f'{EMOJI} Kernel exported to: {output_file} in {running_time} seconds {EMOJI}')


@main.command()
@click.option(
    '-n', '--network',
    help='Path to the network graph or kernel',
    required=True,
    type=click.Path(exists=True, dir_okay=False)
)
@click.option(
    '-i', '--input',
    help='Input data',
    required=True,
    type=click.Path(exists=True, dir_okay=False)
)
@click.option(
    '-o', '--output',
    help='Output path for the results',
    default=OUTPUT,
    show_default=True,
    type=click.Path(exists=True, file_okay=False)
)
@click.option(
    '-m', '--method',
    help='Difussion method',
    type=click.Choice(METHODS),
    required=True,
)
def diffuse(
        network: str,
        input: str,
        output: str,
        method: str,
):
    """Run a diffusion method over a network or pregenerated kernel."""
    raise NotImplementedError
    # TODO : Process arguments and call diffuse


if __name__ == '__main__':
    main()

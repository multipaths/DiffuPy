# -*- coding: utf-8 -*-

"""Command line interface for diffuPy."""

import logging
import os
import pickle
import time

import click
import networkx as nx
import pybel

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
@click.option('--isolates', is_flag=False, help='Include isolates')
@click.option('-l', '--log', is_flag=True, help='Activate debug mode')
def kernel(
        network: str,
        output: str = OUTPUT,
        isolates: bool = None,
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

    # TODO Here goes a function that loads a graph using pandas from csv file and returns a networkx object
    # Temporary is used the PyBEL import function
    graph = pybel.from_pickle(network)

    if isolates:
        click.echo(f'Removing {nx.number_of_isolates(graph)} isolated nodes')
        graph.remove_nodes_from({
            node
            for node in nx.isolates(graph)
        })

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

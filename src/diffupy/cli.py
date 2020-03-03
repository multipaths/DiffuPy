# -*- coding: utf-8 -*-

"""Command line interface for diffuPy."""

import logging
import os
import pickle
import time

import click
import networkx as nx
import pybel

from .constants import OUTPUT
from .kernels import regularised_laplacian_kernel

logger = logging.getLogger(__name__)


@click.group(help='DiffuPy')
def main():
    """Command line interface for diffuPy."""
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")


@main.command()
@click.option(
    '-p', '--path',
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
        path: str,
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

    click.echo(f'Loading graph from {path}')

    # TODO Here goes a function that loads a graph using pandas from csv file and returns a networkx object
    # Temporary is used the PyBEL import function
    graph = pybel.from_pickle(path)

    if isolates:
        click.echo(f'Removing {nx.number_of_isolates(graph)} isolated nodes')
        graph.remove_nodes_from({
            node
            for node in nx.isolates(graph)
        })

    click.echo("Calculating regulatised Laplacian kernel. This might take a while...")

    exe_t_0 = time.time()
    background_mat = regularised_laplacian_kernel(graph)
    exe_t_f = time.time()

    output_file = os.path.join(output, f'{path.split("/")[-1]}.pickle')

    # Export numpy array
    with open(output_file, 'wb') as file:
        pickle.dump(background_mat, file, protocol=4)

    running_time = exe_t_f - exe_t_0

    click.echo(f'Kernel exported to: {output_file} in {running_time} seconds"')


@main.command()
@click.option(
    '-p', '--path',
    help='Path to the pregenerated pickle',
    required=True,
    type=click.Path(exists=True, dir_okay=False)
)
@click.option(
    '-o', '--output',
    help='Output path for results',
    default=OUTPUT,
    show_default=True,
    type=click.Path(exists=True, file_okay=False)
)
def run(
        path: str,
        output: str,
):
    """Run a diffusion method over a pregenerated kernel."""
    raise NotImplementedError


if __name__ == '__main__':
    main()

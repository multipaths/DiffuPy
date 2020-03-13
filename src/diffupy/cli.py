# -*- coding: utf-8 -*-

"""Command line interface for diffuPy."""

import json
import logging
import os
import pickle
import sys
import time

import click

from .constants import OUTPUT, METHODS, EMOJI
from .diffuse import diffuse as run_diffusion
from .kernels import regularised_laplacian_kernel
from .utils import process_network_from_cli, _process_input

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
    """Generate a kernel for a given network."""
    # Configure logging level
    if log:
        logging.basicConfig(level=logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
        logger.setLevel(logging.INFO)

    click.secho(f'{EMOJI} Loading graph from {network} {EMOJI}')

    graph = process_network_from_cli(network)

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
    type=click.File('w'),
    help="Output file",
    default=sys.stdout,
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
    """Run a diffusion method over a network or pre-generated kernel."""
    click.secho(f'{EMOJI} Loading graph from {network} {EMOJI}')
    graph = process_network_from_cli(network)

    click.secho(
        f'{EMOJI} Graph loaded with: \n'
        f'{graph.number_of_nodes()} nodes\n'
        f'{graph.number_of_edges()} edges\n'
        f'{EMOJI}'
    )

    input_scores = _process_input(input)

    results = run_diffusion(
        input_scores,
        method,
        graph,
    )

    json.dump(results, output, indent=2)


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-

"""Command line interface for diffuPy."""

import json
import logging
import os
import pickle
import time

import click

from .process_network import get_kernel_from_network_path

from .constants import OUTPUT, METHODS, EMOJI, RAW, CSV, JSON
from .diffuse import diffuse as run_diffusion
from .kernels import regularised_laplacian_kernel
from .process_input import process_map_and_format_input_data_for_diff
from .process_network import process_graph_from_file

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
        graph: str,
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

    click.secho(f'{EMOJI} Loading graph from {graph} {EMOJI}')

    graph = process_graph_from_file(graph)

    click.secho(f'{EMOJI} Generating regularized Laplacian kernel from graph. This might take a while... {EMOJI}')
    exe_t_0 = time.time()
    kernel = regularised_laplacian_kernel(graph)
    exe_t_f = time.time()

    output_file = os.path.join(output, f'{graph.split("/")[-1]}.pickle')

    # Export numpy array
    with open(output_file, 'wb') as file:
        pickle.dump(kernel, file, protocol=4)

    running_time = exe_t_f - exe_t_0

    click.secho(f'{EMOJI} Kernel exported to: {output_file} in {running_time} seconds {EMOJI}')


@main.command()
@click.option(
    '-i', '--input',
    help='Input data',
    required=True,
    type=click.Path(exists=True, dir_okay=False)
)
@click.option(
    '-n', '--network',
    help='Path to the network graph or kernel',
    required=True,
    type=click.Path(exists=True, dir_okay=False)
)
@click.option(
    '-o', '--output',
    type=click.File('w'),
    help="Output file",
    default=OUTPUT,
)
@click.option(
    '-m', '--method',
    help='Diffusion method',
    type=click.Choice(METHODS),
    default=RAW,
)
@click.option(
    '-b', '--binarize',
    help='If logFC provided in dataset, convert logFC to binary (e.g., up-regulated entities to 1, down-regulated to '
         '-1). For scoring methods that accept quantitative values (i.e., raw & z), node labels can also be codified '
         'with LogFC (in this case, set binarize==False).',
    type=bool,
    default=False,
    show_default=False,
)
@click.option(
    '-t', '--threshold',
    help='Codify node labels by applying a threshold to logFC in input.',
    default=None,
    type=float,
)
@click.option(
    '-a', '--absolute_value',
    help='Codify node labels by applying threshold to | logFC | in input. If absolute_value is set to False,'
         'node labels will be signed.',
    type=bool,
    default=False,
    show_default=False,
)
@click.option(
    '-p', '--p_value',
    help='Statistical significance (p-value).',
    type=float,
    default=0.05,
    show_default=True,
)
@click.option(
    '-f', '--output_format',
    help='Statistical significance (p-value).',
    type=float,
    default=CSV,
    show_default=True,
)
def diffuse(
        input: str,
        network: str,
        output: str = OUTPUT,
        method: str = RAW,
        binarize: bool = False,
        threshold: float = None,
        absolute_value: bool = False,
        p_value: float = 0.05,
        output_format: str = CSV
):
    """Run a diffusion method over a network or pre-generated kernel."""
    click.secho(f'{EMOJI} Loading graph from {network} {EMOJI}')

    kernel = get_kernel_from_network_path(network)

    click.secho(f'Processing data input from {input}.')

    input_scores_dict = process_map_and_format_input_data_for_diff(input,
                                                                   kernel,
                                                                   method,
                                                                   binarize,
                                                                   absolute_value,
                                                                   p_value,
                                                                   threshold,
                                                                   )

    click.secho('Computing the diffusion algorithm.')

    results = run_diffusion(
        input_scores_dict,
        method,
        k=kernel
    )

    if output_format is CSV:
        results.to_csv(output)

    elif output_format is JSON:
        json.dump(results, output, indent=2)

    click.secho(f'{EMOJI} Diffusion performed with success. Output located at {output} {EMOJI}')


if __name__ == '__main__':
    main()

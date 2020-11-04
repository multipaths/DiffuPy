# -*- coding: utf-8 -*-

"""Command line interface for diffuPy."""

import json
import logging
import os
import pickle
import time
from typing import Optional, Callable, Union

import click

from .constants import OUTPUT, METHODS, EMOJI, RAW, CSV, JSON
from .diffuse import diffuse as run_diffusion
from .kernels import regularised_laplacian_kernel
from .process_input import process_map_and_format_input_data_for_diff
from .process_network import get_kernel_from_network_path
from .process_network import process_graph_from_file

logger = logging.getLogger(__name__)


@click.group(help='DiffuPy')
def main():
    """Command line interface for diffuPy."""
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")


@main.command()
@click.option(
    '-g', '--graph',
    help='Input network',
    required=True,
    type=click.Path(exists=True, dir_okay=False)
)
@click.option(
    '-o', '--output',
    help='Output path to store the generated kernel pickle',
    default=os.path.join(OUTPUT, 'kernel.json'),
    show_default=True,
    type=click.Path(file_okay=True)
)
@click.option('-l', '--log', is_flag=True, help='Activate debug mode')
def kernel(
        graph: str,
        output: Optional[str] = os.path.join(OUTPUT, 'kernel.json'),
        log: bool = False
):
    """Generate a kernel for a given network.

    :param network: Path to the network as a (NetworkX) graph to be transformed to kernel.
    :param output: Path (with file name) for the generated scores output file. By default '$OUTPUT/diffusion_scores.csv'
    :param log: Logging profiling option.
    """
    # Configure logging level
    if log:
        logging.basicConfig(level=logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
        logger.setLevel(logging.INFO)

    click.secho(f'{EMOJI} Loading graph from {graph} {EMOJI}')

    click.secho(f'{EMOJI} Generating regularized Laplacian kernel from graph. This might take a while... {EMOJI}')
    exe_t_0 = time.time()
    kernel = regularised_laplacian_kernel(process_graph_from_file(graph))
    exe_t_f = time.time()

    # Export numpy array
    with open(output, 'wb') as file:
        pickle.dump(kernel, file, protocol=4)

    running_time = exe_t_f - exe_t_0

    click.secho(f'{EMOJI} Kernel exported to: {output} in {running_time} seconds {EMOJI}')


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
    default=os.path.join(OUTPUT, 'diffusion_scores.csv'),
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
    '-f', '--format_output',
    help='Choose CSV or JSON output scores file format.',
    type=str,
    default=CSV,
    show_default=True,
)
def diffuse(
        input: str,
        network: str,
        output: Optional[str] = os.path.join(OUTPUT, 'diffusion_scores.csv'),
        method: Union[str, Callable] = RAW,
        binarize: Optional[bool] = False,
        threshold: Optional[float] = None,
        absolute_value: Optional[bool] = False,
        p_value: Optional[float] = 0.05,
        format_output: Optional[str] = CSV,
        kernel_method: Optional[Callable] = regularised_laplacian_kernel
):
    """Run a diffusion method for the provided input_scores over a given network.

    :param input: Path to a (miscellaneous format) data input to be processed/formatted.
    :param network: Path to the network as a (NetworkX) graph or as a (diffuPy.Matrix) kernel.
    :param output: Path (with file name) for the generated scores output file. By default '$OUTPUT/diffusion_scores.csv'
    :param method:  Elected method ["raw", "ml", "gm", "ber_s", "ber_p", "mc", "z"] or custom method FUNCTION(network, scores, kargs). By default 'raw'
    :param binarize: If logFC provided in dataset, convert logFC to binary. By default False
    :param threshold: Codify node labels by applying a threshold to logFC in input. By default None
    :param absolute_value: Codify node labels by applying threshold to | logFC | in input. By default False
    :param p_value: Statistical significance. By default 0.05
    :param format_output: Elected output format ["CSV", "JSON"]. By default 'CSV'
    :param kernel_method: Callable method for kernel computation.
    """
    click.secho(f'{EMOJI} Loading graph from {network} {EMOJI}')

    kernel = get_kernel_from_network_path(network, False, kernel_method=kernel_method)

    click.secho(f'{EMOJI} Processing data input from {input}. {EMOJI}')

    formated_input_scores = process_map_and_format_input_data_for_diff(input,
                                                                       kernel,
                                                                       method,
                                                                       binarize,
                                                                       absolute_value,
                                                                       p_value,
                                                                       threshold,
                                                                       )

    click.secho(f'{EMOJI} Computing the diffusion algorithm. {EMOJI}')

    results = run_diffusion(
        formated_input_scores,
        method,
        k=kernel
    )

    if format_output == CSV:
        results.to_csv(output)

    if format_output == JSON:
        json.dump(results, output, indent=2)

    click.secho(f'{EMOJI} Diffusion performed with success. Output located at {output} {EMOJI}\n')


if __name__ == '__main__':
    main()

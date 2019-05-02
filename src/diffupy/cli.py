# -*- coding: utf-8 -*-

"""Command line interface."""

import logging
import os
import time

import click
import pybel
from pathme.constants import PATHME_DIR
import pickle

import networkx as nx

from diffupy.kernels import regularised_laplacian_kernel

log = logging.getLogger(__name__)


@click.group(help='DiffuPy')
def main():
    """Run DiffuPy."""
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")


"""Diffupy"""


@main.command()
@click.option('-g', '--graph_path', help='Input graph path.',
              default=os.path.join(PATHME_DIR, 'universe', 'pathme_universe_bel_graph_no_flatten.bel.pickle'),
              show_default=True)
@click.option('-o', '--output', help='Output kernel pickle.',
              default=os.path.join(PATHME_DIR, 'kernels', 'regularized_kernel_pathme_universe.pickle'),
              show_default=True)
@click.option('--isolates', is_flag=False, help='Include isolates')
def regularized_kernel_from_bel_graph(bel_graph_path, output_path, isolates):
    bel_graph = pybel.from_pickle(bel_graph_path)

    if not isolates:
        click.echo(f'Number of isolates after getting graph: {nx.number_of_isolates(bel_graph_path)}')

        bel_graph.remove_nodes_from({
            node
            for node in nx.isolates(bel_graph)
        })

    click.echo(bel_graph.summary_str())

    then = time.time()
    background_mat = regularised_laplacian_kernel(bel_graph)
    now = time.time()
    click.echo("It took: ", now - then, " seconds")

    output_path = os.path.join(output_path, 'regularised_laplacian_kernel.pickle')

    with open(output_path, 'wb') as file:
        pickle.dump(background_mat, file)


if __name__ == '__main__':
    main()

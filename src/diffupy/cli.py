# -*- coding: utf-8 -*-

"""Command line interface for diffuPy.

(Explanation from PyBEL documentation)[https://github.com/pybel/pybel/blob/master/src/pybel/cli.py]

Why does this file exist, and why not put this in ``__main__``? You might be tempted to import things from ``__main__``
later, but that will cause problems--the code will get executed twice:

- When you run ``python3 -m diffupy`` python will execute``__main__.py`` as a script. That means there won't be any
  ``diffupy.__main__`` in ``sys.modules``.
- When you import __main__ it will get executed again (as a module) because
  there's no ``pybel.__main__`` in ``sys.modules``.

.. seealso:: http://click.pocoo.org/5/setuptools/#setuptools-integration
"""

import logging
import os
import pickle
import time

import click
import networkx as nx
import pybel

from .constants import DATA_DIR, ensure_output_dirs
from .kernels import regularised_laplacian_kernel


logger = logging.getLogger(__name__)


@click.group(help='DiffuPy')
def main():
    """Command line interface for diffuPy."""
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")


@main.command()
@click.option(
    '-g', '--graph',
    help='Input path to the source BEL graph',
    default=os.path.join(DATA_DIR, 'pickles', 'pathme_universe_bel_graph_no_flatten.bel.pickle'),
    show_default=True
)
@click.option(
    '-o', '--output',
    help='Output path to store the generated kernel pickle',
    default=os.path.join(DATA_DIR, 'kernels'),
    show_default=True
)
@click.option('--isolates', is_flag=False, help='Include isolates')
@click.option('-l', '--log', is_flag=True, help='Activate debug mode')
def kernel(graph: str,
           output: str,
           isolates: bool = None,
           log: bool = None):
    """Generates a kernel for a given BEL graph.

    :param graph: Path to retrieve the input/source BEL graph
    :param output: Path to store the output/generated kernel pickle
    :param isolates: Include isolates
    :param log: Activate debug mode
    :return: None
    """
    ensure_output_dirs()

    # Configure logging level
    if log:
        logging.basicConfig(level=logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
        logger.setLevel(logging.INFO)

    click.echo(f'Loading graph from {graph}')

    # TODO: Discuss, should not reference to PyBEL in diffuPy, since it is offered to treat with any kind/format of graphs,
    #  not only biological/BEL graphs.
    # TODO: Should be used import tools from networkX, no referencing to PyBEL to discuss, treat generalized graphs.
    bel_graph = pybel.from_pickle(graph)

    if isolates:
        click.echo(f'Removing {nx.number_of_isolates(graph)} isolated nodes')
        bel_graph.remove_nodes_from({
            node
            for node in nx.isolates(bel_graph)
        })

    click.echo(
        f'##################\n'
        f'Numerical Summary\n'
        f'##################\n'
        f'{bel_graph.summary_str()}'
    )

    exe_t_0 = time.time()
    background_mat = regularised_laplacian_kernel(bel_graph)
    exe_t_f = time.time()

    output = os.path.join(output, 'regularized_kernel_pathme_universe.pickle')

    # Export numpy array
    with open(output, 'wb') as file:
        pickle.dump(background_mat, file, protocol=4)

    running_time = exe_t_f - exe_t_0

    click.echo(f'Kernel exported to: {output} in {running_time} seconds"')


if __name__ == '__main__':
    main()

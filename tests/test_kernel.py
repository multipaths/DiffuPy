# -*- coding: utf-8 -*-

"""Tests checking kernel functions py implementation based on R package computations."""

import logging
import unittest

import networkx as nx
import numpy as np

from diffupy.kernels import (
    commute_time_kernel,
    p_step_kernel,
    inverse_cosine_kernel,
    diffusion_kernel,
    regularised_laplacian_kernel,
)
from diffupy.matrix import Matrix
from tests.constants import *

log = logging.getLogger(__name__)

"""Helper functions for testing."""


def _run_kernel_test(kernel_func, g, validate_matrix_path):
    """Run kernel test."""
    matrix = kernel_func(g)
    v = Matrix.from_csv(validate_matrix_path, DIFFUSION_KERNEL)

    logging.info(' %s  \n %s\n', 'Computed matrix', matrix)
    logging.info(' %s  \n %s\n', 'Test matrix', v)
    # Assert rounded similarity (floating comma)
    assert np.allclose(matrix, v)
    logging.info(' Test ' + kernel_func.__name__ + ' passed')


"""Tests"""


class KernelsTest(unittest.TestCase):
    """Kernel test."""

    graph = nx.read_gml(GML_FILE_EXAMPLE, label='id')

    _run_kernel_test(commute_time_kernel, graph, COMMUTE_TIME_KERNEL)
    _run_kernel_test(diffusion_kernel, graph, DIFFUSION_KERNEL)
    _run_kernel_test(p_step_kernel, graph, P_STEP_KERNEL)
    _run_kernel_test(inverse_cosine_kernel, graph, INVERSE_COSINE_KERNEL)
    _run_kernel_test(regularised_laplacian_kernel, graph, REGULARISED_LAPLACIAN_KERNEL)

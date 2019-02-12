# -*- coding: utf-8 -*-

"""Tests checking kernel functions py implementation based on R package computations."""

import logging
import networkx as nx
import numpy as np
import unittest

from diffupy.diffuse_raw import diffuse_raw
from diffupy.kernel import commute_time_kernel, p_step_kernel, inverse_cosine_kernel, diffusion_kernel, regularised_laplacian_kernel
from diffupy.matrix import Matrix

from .constants import *

log = logging.getLogger(__name__)


class KernelsTest(unittest.TestCase):
    G = nx.read_gml(GML_FILE_EXAMPLE, label='id')

    def run_kernel_test(kernel_func, G, validate_matrix_path):
        M = kernel_func(G)
        V = Matrix.from_csv(validate_matrix_path)

        logging.info(' %s  \n %s\n', 'Computed matrix', M)
        logging.info(' %s  \n %s\n', 'Test matrix', V)
        # Assert rounded similarity (floating comma)
        assert np.allclose(M, V)
        logging.info(' Test ' + kernel_func.__name__ + ' passed')

    run_kernel_test(commute_time_kernel, G, COMMUTE_TIME_KERNEL)
    run_kernel_test(diffusion_kernel, G, DIFFUSION_KERNEL)
    run_kernel_test(p_step_kernel, G, P_STEP_KERNEL)
    run_kernel_test(inverse_cosine_kernel, G, INVERSE_COSINE_KERNEL)
    run_kernel_test(regularised_laplacian_kernel, G, REGULARISED_LAPLACIAN_KERNEL)


class DiffuseRawTest(unittest.TestCase):
    G = nx.read_gml(GML_FILE_EXAMPLE, label='id')

    def run_score_test(score_func, G, input_scores, test_output_scores, z=False):
        input_scores = Matrix.from_csv(input_scores)
        test_output_scores = Matrix.from_csv(test_output_scores)

        computed_output_scores = score_func(G, input_scores, z)

        if isinstance(computed_output_scores, Matrix):
            computed_output_scores = computed_output_scores.mat

        if isinstance(test_output_scores, Matrix):
            test_output_scores = test_output_scores.mat

        logging.info(' %s  \n %s\n', 'Computed matrix', computed_output_scores)
        logging.info(' %s  \n %s\n', 'Test matrix', test_output_scores)
        # Assert rounded similarity (floating comma)
        assert np.allclose(computed_output_scores, test_output_scores)
        logging.info(' Test ' + score_func.__name__ + ' passed')

    run_score_test(diffuse_raw, G, INPUT_SCORES, OUTPUT_SCORES)
    run_score_test(diffuse_raw, G, INPUT_SCORES, OUTPUT_Z_SCORES, z=True)

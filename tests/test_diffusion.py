# -*- coding: utf-8 -*-

"""Tests checking kernel functions py implementation based on R package computations."""

import logging
import unittest

import networkx as nx
import numpy as np

from diffupy.diffuse import diffuse
from diffupy.matrix import Matrix
from tests.constants import *

log = logging.getLogger(__name__)

"""Helper functions for testing"""


def _run_diffusion_method_test(method, g, input_scores, test_output_scores):
    """Help for test class."""
    computed_output_scores = diffuse(input_scores, method, graph=g)

    if isinstance(computed_output_scores, Matrix):
        computed_output_scores = computed_output_scores.mat

    if isinstance(test_output_scores, Matrix):
        test_output_scores = test_output_scores.mat

    logging.info(' %s  \n %s\n', 'Computed matrix', computed_output_scores)
    logging.info(' %s  \n %s\n', 'Test matrix', test_output_scores)
    # Assert rounded similarity (floating comma)
    assert np.allclose(computed_output_scores, test_output_scores)
    logging.info(' Test ' + method + ' passed')


"""Tests"""


class DiffuseTest(unittest.TestCase):
    """Test diffusion methods."""

    graph = nx.read_gml(GML_FILE_EXAMPLE, label='id')

    _run_diffusion_method_test('raw', graph, INPUT_SCORES, OUTPUT_RAW_SCORES)
    _run_diffusion_method_test('z', graph, INPUT_SCORES, OUTPUT_Z_SCORES)
    _run_diffusion_method_test('ml', graph, INPUT_SCORES, OUTPUT_ML_SCORES)
    _run_diffusion_method_test('gm', graph, INPUT_UNLABELED_SCORES, OUTPUT_GM_SCORES)

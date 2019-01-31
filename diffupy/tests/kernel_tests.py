# -*- coding: utf-8 -*-

"""Tests checking kernel functions py implementation based on R package computations."""


import os
import numpy as np
import logging
import networkx as nx
from diffupy.src.diffupy.kernels import commute_time_kernel, p_step_kernel, inverse_cosine_kernel, diffusion_kernel

TEST_FOLDER = os.path.dirname(os.path.realpath(__file__))
resources_path = os.path.join(TEST_FOLDER, 'resources')

# Kernel tests: 04_unit_testing
# Graphs
G1 = os.path.join(TEST_FOLDER, '_graph.gml')

# Validation matrix
commtK_path = os.path.join(TEST_FOLDER, 'commuteTimeKernel.csv')
pstepK_path = os.path.join(TEST_FOLDER, 'pStepKernel.csv')
invcosK_path = os.path.join(TEST_FOLDER, 'inverseCosineKernel.csv')
diffuK_path = os.path.join(TEST_FOLDER, 'diffusionKernel.csv')


log = logging.getLogger(__name__)


def csv_labeled_matrix_to_nparray(path):
    # Import matrix from csv file and remove headers
    m = np.genfromtxt(path, delimiter=',')
    return np.array([[x for x in a if ~np.isnan(x)] for a in m[1:]])


class KernelsTest():

    G = nx.read_gml(G1, label='id')

    def run_kernel_test(self, kernel_func, G, validate_matrix_path):
        M = kernel_func(G)
        V = csv_labeled_matrix_to_nparray(validate_matrix_path)

        logging.info(' %s  \n %s\n', 'Computed matrix', M)
        logging.info(' %s  \n %s\n', 'Test matrix', V)
        # Assert rounded similarity (floating comma)
        assert np.allclose(M, V)
        logging.info(' Test ' + kernel_func.__name__ + ' passed')

    run_kernel_test(commute_time_kernel, G, commtK_path)
    run_kernel_test(p_step_kernel, G, pstepK_path)
    run_kernel_test(inverse_cosine_kernel, G, invcosK_path)
    run_kernel_test(diffusion_kernel, G, diffuK_path)
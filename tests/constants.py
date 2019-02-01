# -*- coding: utf-8 -*-

"""Tests constants."""
import os

TEST_FOLDER = os.path.dirname(os.path.realpath(__file__))
RESOURCES_FOLDER = os.path.join(TEST_FOLDER, 'resources')

GML_FILE_EXAMPLE = os.path.join(RESOURCES_FOLDER, '_graph.gml')
COMMUTE_TIME_KERNEL = os.path.join(RESOURCES_FOLDER, 'commuteTimeKernel.csv')
DIFFUSION_KERNEL = os.path.join(RESOURCES_FOLDER, 'diffusionKernel.csv')
INVERSE_COSINE_KERNEL = os.path.join(RESOURCES_FOLDER, 'inverseCosineKernel.csv')
P_STEP_KERNEL = os.path.join(RESOURCES_FOLDER, 'pStepKernel.csv')
REGULARISED_LAPLACIAN_KERNEL = os.path.join(RESOURCES_FOLDER, 'regulatisedLaplicianKernel.csv')

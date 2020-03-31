# -*- coding: utf-8 -*-

"""Tests validator module."""

import logging
import unittest

from diffupy.constants import *
from diffupy.matrix import Matrix
from diffupy.process_input import process_input
from diffupy.validate_input import _validate_scores
from tests.constants import *

log = logging.getLogger(__name__)

original = {'A': 0.7, 'B': 1.2, 'C': -0.2, 'D': -0.4, 'E': -2.2}


class ValidateTest(unittest.TestCase):
    """Test validation of results."""

    def test_quantitative_bin_id(self):
        """Test codify input for quantitative scoring methods- only entity IDs given (binary labels)"""
        input = NODE_TEST_PATH
        input_scores_dict = process_input(
            input, method=RAW, binning=True, absolute_value=True, p_value=0.05, threshold=None)

        self.assertEqual(input_scores_dict, {'A': 1, 'B': 1, 'C': 1, 'D': 1, 'E': 1})

    def test_quantitative_bin_fc_sign(self):
        """Test codify input for quantitative scoring methods- logFC given (binary, signed labels)"""
        input = NODE_LOGFC_TEST_PATH
        input_scores_dict = process_input(
            input, method=RAW, binning=True, absolute_value=False, p_value=0.05, threshold=0.5)

    def test_quantitative_bin_fc_abs(self):
        """Test codify input for quantitative scoring methods- logFC given (binary, absolute values)"""
        input = NODE_LOGFC_TEST_PATH
        input_scores_dict = process_input(
            input, method=RAW, binning=True, absolute_value=True, p_value=0.05, threshold=0.5)

    def test_quantitative_bin_fcp_sign(self):
        """Test codify input for quantitative scoring methods- logFC and adj. p-value given (binary, signed labels)"""
        input = NODE_LOGFC_PVAL_TEST_PATH
        input_scores_dict = process_input(
            input, method=RAW, binning=True, absolute_value=False, p_value=0.05, threshold=0.5)

    def test_quantitative_bin_fcp_abs(self):
        """Test codify input for quantitative scoring methods- logFC and adj. p-value given (binary, absolute values)"""
        input = NODE_LOGFC_PVAL_TEST_PATH
        input_scores_dict = process_input(
            input, method=RAW, binning=True, absolute_value=True, p_value=0.05, threshold=0.5)

    def test_quantitative_fc_sign(self):
        """Test codify input for quantitative scoring methods- logFC given (quantitative, signed labels)"""
        input = NODE_LOGFC_TEST_PATH
        input_scores_dict = process_input(
            input, method=RAW, binning=False, absolute_value=False, p_value=0.05, threshold=0.5)

    def test_quantitative_fc_abs(self):
        """Test codify input for quantitative scoring methods- logFC given (quant., absolute values)"""
        input = NODE_LOGFC_TEST_PATH
        input_scores_dict = process_input(
            input, method=RAW, binning=False, absolute_value=True, p_value=0.05, threshold=0.5)

    def test_quantitative_fcp_sign(self):
        """Test codify input for quantitative scoring methods- logFC and adj. p-value given (quant., signed labels)"""
        input = NODE_LOGFC_PVAL_TEST_PATH
        input_scores_dict = process_input(
            input, method=RAW, binning=False, absolute_value=False, p_value=0.05, threshold=0.5)

    def test_quantitative_fcp_abs(self):
        """Test codify input for quantitative scoring methods- logFC and adj. p-value given (quant., absolute values)"""
        input = NODE_LOGFC_PVAL_TEST_PATH
        input_scores_dict = process_input(
            input, method=RAW, binning=False, absolute_value=True, p_value=0.05, threshold=0.5)

    def test_non_quantitative_bin_id(self):
        """Test codify input for non-quantitative scoring methods- only entity IDs given (binary labels)"""
        input = NODE_TEST_PATH
        input_scores_dict = process_input(
            input, method=ML, binning=True, absolute_value=True, p_value=0.05, threshold=0.5)

    def test_non_quantitative_bin_fc_abs(self):
        """Test codify input for non-quantitative scoring methods- logFC given (binary, absolute values)"""
        input = NODE_LOGFC_TEST_PATH
        input_scores_dict = process_input(
            input, method=ML, binning=True, absolute_value=True, p_value=0.05, threshold=0.5)

    def test_non_quantitative_bin_fcp_abs(self):
        """Test codify input for non-quant. scoring methods- logFC and adj. p-value given (binary, absolute values)"""
        input = NODE_LOGFC_PVAL_TEST_PATH
        input_scores_dict = process_input(
            input, method=ML, binning=True, absolute_value=True, p_value=0.05, threshold=0.5)

    def test_validate_scores_1(self):
        """Test validate scores 1."""
        matrix = Matrix([1, 2, 3, 4], name='Test Matrix')

        _validate_scores(matrix)

    def test_validate_scores_2(self):
        """Test validate scores 2."""
        matrix = Matrix(
            [1, 2, 3, 4],
            cols_labels=['1', '2', '3', '4'],
            rows_labels=['1', '2', '3', '4'],
            name='Test Matrix 2'
        )

        _validate_scores(matrix)

    def test_validate_scores_3(self):
        """One score in the array is not numeric."""
        matrix = Matrix(
            [1, '2', 3, 4],
            cols_labels=['1', '2', '3', '4'],
            rows_labels=['1', '2', '3', '4'],
            name='Test Matrix 3'
        )
        with self.assertRaises(ValueError):
            _validate_scores(matrix)

    def test_validate_scores_4(self):
        """Test empty matrix."""
        matrix = Matrix(
            None,
            cols_labels=[None],
            rows_labels=[None],
            name='Test Matrix 4'
        )
        with self.assertRaises(ValueError):
            _validate_scores(matrix)

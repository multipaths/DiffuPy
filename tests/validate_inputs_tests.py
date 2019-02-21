# -*- coding: utf-8 -*-

"""Tests validator module."""

import logging
import unittest

from diffuPy.matrix import Matrix
from diffuPy.validate_inputs import _validate_scores

log = logging.getLogger(__name__)


class ValidateTest(unittest.TestCase):

    def test_validate_scores_1(self):
        matrix = Matrix([1, 2, 3, 4], name='Test Matrix')

        _validate_scores(matrix)

    def test_validate_scores_2(self):
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
            cols_labels=['1', '2', '3', '4']
            , rows_labels=['1', '2', '3', '4'],
            name='Test Matrix 3'
        )
        with self.assertRaises(ValueError): _validate_scores(matrix)


    def test_validate_scores_4(self):
        """Test empty matrix."""
        matrix = Matrix(
            None,
            cols_labels=[None],
            rows_labels=[None],
            name='Test Matrix 4'
        )
        with self.assertRaises(ValueError): _validate_scores(matrix)

# -*- coding: utf-8 -*-

"""Tests validator module."""

import logging
import unittest

from diffupy.constants import *
from diffupy.matrix import Matrix
from diffupy.process_input import process_input_data, map_labels_input, \
    format_input_for_diffusion
from diffupy.process_network import get_graph_from_df
from diffupy.validate_input import _validate_scores

from .constants import *

log = logging.getLogger(__name__)


class ValidateTest(unittest.TestCase):
    """Test validation of results."""

    def test_quantitative_bin_id(self):
        """Test codify label_input for quantitative scoring methods- only entity IDs given (binary labels)."""
        input = NODE_TYPE_COL_TEST_PATH
        input_labels_dict = process_input_data(
            input, method=RAW, binning=True, absolute_value=True, p_value=0.05, threshold=None,
        )
        self.assertEqual(input_labels_dict, {'Metabolite': {'C': 1}, 'Gene': {'A': 1, 'B': 1, 'D': 1, 'E': 1}})

    def test_quantitative_bin_fc_sign(self):
        """Test codify label_input for quantitative scoring methods- logFC given (binary, signed labels)."""
        input = NODE_LOGFC_TEST_PATH
        input_labels_dict = process_input_data(
            input, method=RAW, binning=True, absolute_value=False, p_value=0.05, threshold=0.5,
        )
        self.assertEqual(input_labels_dict, {'A': 1, 'B': 1, 'C': 0, 'D': 0, 'E': -1})

    def test_quantitative_bin_fc_abs(self):
        """Test codify label_input for quantitative scoring methods- logFC given (binary, absolute values)."""
        input = NODE_LOGFC_TEST_PATH
        input_labels_dict = process_input_data(
            input, method=RAW, binning=True, absolute_value=True, p_value=0.05, threshold=0.5,
        )
        self.assertEqual(input_labels_dict, {'A': 1, 'B': 1, 'C': 0, 'D': 0, 'E': 1})

    def test_quantitative_bin_fcp_sign(self):
        """Test codify label_input for quantitative scoring methods- logFC and adj. p-value given (binary, signed labels)."""
        input = NODE_LOGFC_PVAL_TEST_PATH
        input_labels_dict = process_input_data(
            input, method=RAW, binning=True, absolute_value=False, p_value=0.05, threshold=0.5,
        )
        self.assertEqual(input_labels_dict, {'A': 0, 'B': 1, 'C': 0, 'D': 0, 'E': -1})

    def test_quantitative_bin_fcp_abs(self):
        """Test codify label_input for quant. scoring methods- logFC and adj. p-value given (binary, absolute values)."""
        input = NODE_LOGFC_PVAL_TEST_PATH
        input_labels_dict = process_input_data(
            input, method=RAW, binning=True, absolute_value=True, p_value=0.05, threshold=0.5,
        )
        self.assertEqual(input_labels_dict, {'A': 0, 'B': 1, 'C': 0, 'D': 0, 'E': 1})

    def test_quantitative_fc_sign(self):
        """Test codify label_input for quantitative scoring methods- logFC given (quantitative, signed labels)."""
        input = NODE_LOGFC_TEST_PATH
        input_labels_dict = process_input_data(
            input, method=RAW, binning=False, absolute_value=False, p_value=0.05, threshold=0.5,
        )
        self.assertEqual(input_labels_dict, {'A': 0.7, 'B': 1.2, 'C': 0, 'D': 0, 'E': -2.2})

    def test_quantitative_fc_abs(self):
        """Test codify label_input for quantitative scoring methods- logFC given (quant., absolute values)."""
        input = NODE_LOGFC_TEST_PATH
        input_labels_dict = process_input_data(
            input, method=RAW, binning=False, absolute_value=True, p_value=0.05, threshold=0.5,
        )
        self.assertEqual(input_labels_dict, {'A': 0.7, 'B': 1.2, 'C': 0, 'D': 0, 'E': 2.2})

    def test_quantitative_fcp_sign(self):
        """Test codify label_input for quantitative scoring methods- logFC and adj. p-value given (quant., signed labels)."""
        input = NODE_LOGFC_PVAL_TEST_PATH
        input_labels_dict = process_input_data(
            input, method=RAW, binning=False, absolute_value=False, p_value=0.05, threshold=0.5,
        )
        self.assertEqual(input_labels_dict, {'A': 0, 'B': 1.2, 'C': 0, 'D': 0, 'E': -2.2})

    def test_quantitative_fcp_abs(self):
        """Test codify label_input for quant. scoring methods- logFC and adj. p-value given (quant., absolute values)."""
        input = NODE_LOGFC_PVAL_TEST_PATH
        input_labels_dict = process_input_data(
            input, method=RAW, binning=False, absolute_value=True, p_value=0.05, threshold=0.5,
        )
        self.assertEqual(input_labels_dict, {'A': 0, 'B': 1.2, 'C': 0, 'D': 0, 'E': 2.2})

    def test_non_quantitative_bin_id(self):
        """Test codify label_input for non-quantitative scoring methods- only entity IDs given (binary labels)."""
        input = NODE_TYPE_COL_TEST_PATH
        input_labels_dict = process_input_data(
            input, method=ML, binning=True, absolute_value=True, p_value=0.05, threshold=None,
        )
        self.assertEqual(input_labels_dict, {'Metabolite': {'C': 1}, 'Gene': {'A': 1, 'B': 1, 'D': 1, 'E': 1}})

    def test_non_quantitative_bin_fc_abs(self):
        """Test codify label_input for non-quantitative scoring methods- logFC given (binary, absolute values (sign))."""
        input = NODE_LOGFC_TEST_PATH
        input_labels_dict = process_input_data(
            input, method=ML, binning=True, absolute_value=True, p_value=0.05, threshold=0.5,
        )
        self.assertEqual(input_labels_dict, {'A': 1, 'B': 1, 'C': -1, 'D': -1, 'E': 1})

    def test_non_quantitative_bin_fcp_abs(self):
        """Test codify label_input for non-quant. scoring methods- logFC and adj. p-value given (binary, absolute values)."""
        input = NODE_LOGFC_PVAL_TEST_PATH
        input_labels_dict = process_input_data(
            input, method=ML, binning=True, absolute_value=True, p_value=0.05, threshold=0.5,
        )
        self.assertEqual(input_labels_dict, {'A': -1, 'B': 1, 'C': -1, 'D': -1, 'E': 1})

    def test_map_labels_input_label_list_background_list(self):
        """Test map label_input."""
        mapping = map_labels_input(input_labels=['A', 'B', 'C', 'D'],
                                   background_labels=['A', 'B', 'C'])
        # As set because the order is not relevant.
        self.assertEqual(set(mapping), {'A', 'C', 'B'})

    def test_map_labels_input_label_list_background_dict(self):
        """Test map label_input."""
        mapping = map_labels_input(input_labels=['A', 'B', 'C', 'D'],
                                   background_labels={'Gene': ['A', 'B'], 'Metabolite': ['C']})

        self.assertEqual(mapping, {'Gene': ['A', 'B'], 'Metabolite': ['C']})

    def test_map_labels_input_type_dict_label_list_background_list(self):
        """Test map label_input."""
        mapping = map_labels_input(input_labels={'Gene': ['A', 'B'], 'Metabolite': ['C', 'D']},
                                   background_labels=['A', 'B', 'C'])

        self.assertEqual(mapping, {'Gene': ['A', 'B'], 'Metabolite': ['C']})

    def test_map_labels_input_type_dict_label_dict_background_dict(self):
        """Test map label_input."""
        # If the labels are classified in another type ('D' and 'B'), since it do not match with the background it will be not mapped.
        mapping = map_labels_input(input_labels={'Gene': ['A'], 'Metabolite': ['C', 'B']},
                                   background_labels={'Gene': ['A', 'B'], 'Metabolite': ['D']})

        self.assertEqual(mapping, {'Gene': ['A']})

    def test_map_labels_input_label_scores_dict_background_list(self):
        """Test map label_input."""
        mapping = map_labels_input(input_labels={'A': 1, 'B': 1, 'D': 1, 'E': 1},
                                   background_labels=['B', 'C', 'D'])

        self.assertEqual(mapping, {'B': 1, 'D': 1})

    def test_map_labels_input_label_scores_dict_background_dict(self):
        """Test map label_input."""
        mapping = map_labels_input(input_labels={'A': 1, 'B': 1, 'D': 1, 'E': 1},
                                   background_labels={'Gene': ['A', 'B'], 'Metabolite': ['D']})

        self.assertEqual(mapping, {'Metabolite': {'D': 1}, 'Gene': {'A': 1, 'B': 1}})

    def test_map_labels_input_type_dict_label_scores_dict_background_list(self):
        """Test map label_input."""
        mapping = map_labels_input(input_labels={'Metabolite': {'C': 1}, 'Gene': {'A': 1, 'B': 1, 'D': 1, 'E': 1}},
                                   background_labels=['A', 'B', 'C'])

        self.assertEqual(mapping, {'Metabolite': {'C': 1}, 'Gene': {'A': 1, 'B': 1}})

    def test_map_labels_input_type_dict_label_scores_dict_background_dict(self):
        """Test map label_input."""
        # If the labels are classified in another type ('D' and 'B'), since it do not match with the background it will be not mapped.
        mapping = map_labels_input(input_labels={'Metabolite': {'C': -1}, 'Gene': {'A': 1, 'B': 1, 'D': 1, 'E': 1}},
                                   background_labels={'Gene': ['A', 'B'], 'Metabolite': ['C']})

        self.assertEqual(mapping, {'Metabolite': {'C': -1}, 'Gene': {'A': 1, 'B': 1}})

    def test_map_labels_input_type_dict_label_scores_dict_background_two_dimensional_dict(self):
        """Test map label_input."""
        # If the labels are classified in another type ('D' and 'B'), since it do not match with the background it will be not mapped.
        mapping = map_labels_input(input_labels={'Metabolite': {'C': -1}, 'Gene': {'A': 1, 'B': 1, 'D': 1, 'E': 1}},
                                   background_labels={'db1': {'Gene': ['A', 'B']}, 'db2': {'Metabolite': ['C']}},
                                   show_descriptive_stat=True)

    def test_network(self):
        """Test generate graph from csv."""
        graph = get_graph_from_df(NETWORK_PATH, CSV)
        graph_nodes = set(graph.nodes())
        graph_edges = set(graph.edges())

        self.assertEqual(graph_nodes, {'A', 'B', 'C', 'D', 'E', 'F', 'G'})
        self.assertEqual(graph_edges, {
            ('A', 'B'),
            ('A', 'D'),
            ('B', 'C'),
            ('B', 'G'),
            ('C', 'D'),
            ('C', 'E'),
            ('C', 'F'),
            ('D', 'E'),
            ('E', 'F'),
            ('G', 'F'),
        })

    def test_node_mapping(self):
        """Test mapping of nodes in label_input to nodes in network."""
        input = NODE_LOGFC_TEST_PATH
        input_labels_dict = process_input_data(
            input, method=RAW, binning=False, absolute_value=True, p_value=0.05, threshold=0.5,
        )

        graph = get_graph_from_df(NETWORK_PATH, CSV)
        graph_nodes = list(graph.nodes())

        mapped_nodes_list = map_labels_input(input_labels_dict, graph_nodes)

        self.assertEqual(mapped_nodes_list, {'A': 0.7, 'B': 1.2, 'C': 0.0, 'D': 0.0, 'E': 2.2})

    def test_validate_scores_1(self):
        """Test validate scores 1."""
        matrix = Matrix([[1, 2, 3, 4]],
                        rows_labels=['1'],
                        cols_labels=['1', '2', '3', '4'],
                        name='Test Matrix')
        _validate_scores(matrix)

    def test_validate_scores_2(self):
        """Test validate scores 2."""
        matrix = Matrix(
            [[1], [2]],
            rows_labels=['1', '2'],
            cols_labels=['1'],
            name='Test Matrix 2'
        )
        _validate_scores(matrix)

    def test_validate_scores_3(self):
        """One score in the array is not numeric."""
        matrix = Matrix(
            [[1, 2, 3, 4]],
            cols_labels=['1', '2', '3', '4'],
            rows_labels=['1', '2'],
            name='Test Matrix 3'
        )
        with self.assertRaises(IndexError):
            _validate_scores(matrix)

    kernel_test_1 = Matrix(
        [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
        cols_labels=['A', 'B', 'C', 'D'],
        rows_labels=['A', 'B', 'C', 'D'],
        name='Test Kernel 1'
    )

    kernel_test_2 = Matrix(
        [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
        cols_labels=['A', 'B', 'C', 'F'],
        rows_labels=['A', 'B', 'C', 'F'],
        name='Test Kernel 2'
    )

    kernel_test_3 = Matrix(
        [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
        cols_labels=['A', 'B', 'C', 'D', 'F'],
        rows_labels=['A', 'B', 'C', 'D', 'F'],
        name='Test Kernel 3'
    )

    def test_format_input_for_diffusion_label_list(self):
        """Test empty matrix."""
        processed_mapped_nodes_list = format_input_for_diffusion(
            map_labels_input({'Metabolite': {'C': -1}, 'Gene': {'A': 2, 'B': 1}, 'mirnas': {'A': 1, 'T': 1}},
                             self.kernel_test_1.rows_labels),
            self.kernel_test_1,
        )

        # TODO: Implement in Matrix equal, now if the col order is mixed it raises error
        # assert(np.allclose(processed_mapped_nodes_list.mat,
        #                    np.array([[-1, 2, 1],
        #                              [-1, 1, -1],
        #                              [-1, -1, -1],
        #                              [-1, -1, -1]]
        #                             )
        #                    )
        #        )
        # self.assertEqual(processed_mapped_nodes_list.cols_labels,
        #                 ['Metabolite', 'Gene', 'mirnas']
        #                 )
        # self.assertEqual(processed_mapped_nodes_list.rows_labels,
        #                 ['A', 'B', 'C', 'D']
        #                 )

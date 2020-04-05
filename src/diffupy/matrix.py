# -*- coding: utf-8 -*-

"""Main Matrix Class."""

import logging
import os

import numpy as np
import pandas as pd

from .utils import get_label_ix_mapping, get_label_list_graph, get_laplacian, decode_labels, get_idx_scores_mapping, \
    get_repeated_labels

log = logging.getLogger(__name__)

__all__ = [
    'Matrix',
    'LaplacianMatrix',
]


class Matrix:
    """Matrix class."""

    def __init__(
        self,
        mat=None,
        rows_labels=None,
        cols_labels=None,
        graph=None,
        quadratic=False,
        name='',
        init_value=None,
        **kwargs
    ):
        """Initialize matrix.

        :param mat: matrix initialization
        :param rows_labels:
        :param cols_labels: column labels
        :param graph: graph
        :param quadratic: quadratic
        :param name: name
        :param init_value: value to be initialized (int) or list of values from labels
        """
        # This initialization would make a matrix representing the input scores
        if isinstance(rows_labels, list) or isinstance(rows_labels, set) or isinstance(rows_labels, np.ndarray):
            self.rows_labels = list(rows_labels)
        # This initialization would make a matrix representing the graph (taken the graph labels 'name')
        elif graph:
            self.rows_labels = list(get_label_list_graph(graph, 'name'))
        else:
            raise ValueError('No rows_labels list provided.')

        if isinstance(cols_labels, list) or isinstance(cols_labels, set) or isinstance(cols_labels, np.ndarray):
            self._cols_labels = list(cols_labels)

        elif not quadratic:
            raise ValueError('No cols_labels list provided.')

        self.name = name
        self.quadratic = quadratic

        # if isinstance(init_value, list):
        #     mat = init_value
        # todo: THIS SHOULD BE AN ELIF
        if init_value is not None and self.rows_labels and list(self.cols_labels):
            mat = np.full((len(self.rows_labels), len(self.cols_labels)), init_value)

        elif not list(mat):
            raise ValueError('An path matrix or initialization should be provided.')

        self.mat = np.array(mat)

        self.get_labels = True
        self.get_indices = False

        self.validate_labels()

    def __str__(self):
        """Return a string representation of the Matrix."""
        s = f"        {self.cols_labels}"

        for i, row_label in enumerate(self.rows_labels):
            s += f"\n {row_label}  {self.mat[i]} "

        return f"\nmatrix {self.name} \n  {s} \n "

    def __iter__(self, **kargs):
        """Help method for the iteration of the Matrix."""
        self.i = -1
        self.j = 0

        if 'get_indices' in kargs:
            self.get_indices = kargs['get_indices']
        if 'get_labels' in kargs:
            self.get_labels = kargs['get_labels']

        return self

    def __next__(self):
        """Help method for the iteration of the Matrix."""
        if self.i >= len(self.rows_labels) - 1 and self.j >= len(self.cols_labels) - 1:
            self.get_labels = True
            self.get_indices = False
            raise StopIteration

        if self.i >= len(self.rows_labels) - 1:
            self.i = 0
            self.j += 1
        else:
            self.i += 1

        nxt = tuple()
        if len(self.rows_labels) == 1:
            nxt += (self.mat[self.j],)
        else:
            nxt += (self.mat[self.i][self.j],)

        if self.get_indices:
            nxt += (self.i, self.j,)

        if self.get_labels:
            nxt += (self.rows_labels[self.i], self.cols_labels[self.j])

        return nxt

    def __copy__(self):
        """Return a copy of a Matrix object."""
        return Matrix(self.mat,
                      rows_labels=self.rows_labels,
                      cols_labels=self.cols_labels,
                      name=self.name,
                      quadratic=self.quadratic,
                      )

    """Validators """

    def validate_labels(self):
        """Sanity function to check the dimensionality of the Matrix."""
        if self.rows_labels:
            self.rows_labels = decode_labels(self.rows_labels)
            if len(self.rows_labels) != len(set(self.rows_labels)):
                dup = get_repeated_labels(self.rows_labels)
                raise Exception(
                    'Duplicate row labels in Matrix. /n duplicated number: {} /n duplicated list: {}'.format(
                        len(dup),
                        dup
                    )
                )

        if hasattr(self, '_cols_labels'):
            self._cols_labels = decode_labels(self.cols_labels)
            if len(self._cols_labels) != len(set(self._cols_labels)):
                raise Exception('Duplicate column labels in Matrix.')

    def update_ix_mappings(self):
        """Update the index-label mapping."""
        if hasattr(self, '_rows_labels_ix_mapping') and self.rows_labels:
            self._rows_labels_ix_mapping = get_label_ix_mapping(self.rows_labels)

        if hasattr(self, '_cols_labels_ix_mapping') and hasattr(self, '_cols_labels'):
            self._cols_labels_ix_mapping = get_label_ix_mapping(self._cols_labels)

    def validate_labels_and_update_ix_mappings(self):
        """Update function, called when the Matrix mutates, combining the two previous functionalities."""
        self.validate_labels()
        self.update_ix_mappings()

    """Getters and Setters"""

    # Columns labels
    @property
    def cols_labels(self):
        """Return a copy of Matrix Object."""
        if self.quadratic:
            return self.rows_labels

        return self._cols_labels

    @cols_labels.setter
    def cols_labels(self, cols_labels):
        """Set column labels."""
        if self.quadratic:
            self.rows_labels = list(cols_labels)
        else:
            self._cols_labels = list(cols_labels)

    # Rows ix mapping
    @property
    def rows_labels_ix_mapping(self):
        """Set row labels to ix."""
        if hasattr(self, '_rows_labels_ix_mapping'):
            return self._rows_labels_ix_mapping

        self._rows_labels_ix_mapping = get_label_ix_mapping(self.rows_labels)
        return self._rows_labels_ix_mapping

    @rows_labels_ix_mapping.setter
    def rows_labels_ix_mapping(self, rows_labels_ix_mapping):
        """Set labels labels to ix."""
        self._rows_labels_ix_mapping = rows_labels_ix_mapping

    # Columns ix mapping
    @property
    def cols_labels_ix_mapping(self):
        """Set column labels to ix."""
        if self.quadratic:
            return self.rows_labels_ix_mapping

        if hasattr(self, '_cols_labels_ix_mapping'):
            return self._cols_labels_ix_mapping

        self._cols_labels_ix_mapping = get_label_ix_mapping(self.cols_labels)
        return self._cols_labels_ix_mapping

    @cols_labels_ix_mapping.setter
    def cols_labels_ix_mapping(self, cols_labels_ix_mapping):
        """Set mapping labels to ix."""
        if self.quadratic:
            self._rows_labels_ix_mapping = cols_labels_ix_mapping

        self._cols_labels_ix_mapping = cols_labels_ix_mapping

    # Rows scores mapping
    @property
    def rows_idx_scores_mapping(self):
        """Set mapping indexes to scores."""
        if hasattr(self, '_rows_idx_scores_mapping'):
            return self._rows_idx_scores_mapping

        self._rows_idx_scores_mapping = get_idx_scores_mapping(self.mat)

        return self._rows_idx_scores_mapping

    @rows_idx_scores_mapping.setter
    def rows_idx_scores_mapping(self, rows_idx_scores_mapping):
        """Set mapping rows to ids."""
        self._rows_idx_scores_mapping = rows_idx_scores_mapping

    # Columns scores mapping
    @property
    def cols_idx_scores_mapping(self):
        """Set mapping indexes to scores."""
        if hasattr(self, '_cols_idx_scores_mapping'):
            return self._cols_idx_scores_mapping

        self._cols_idx_scores_mapping = get_idx_scores_mapping(self.mat.transpose())

        return self._cols_idx_scores_mapping

    @cols_idx_scores_mapping.setter
    def cols_idx_scores_mapping(self, cols_idx_scores_mapping):
        if self.quadratic:
            self._rows_idx_scores_mapping = cols_idx_scores_mapping
        self._cols_idx_scores_mapping = cols_idx_scores_mapping

    """Getters setters and delete from labels"""

    def get_row_from_label(self, label):
        """Get row from labels."""
        return self.mat[self.rows_labels_ix_mapping[label]]

    def set_row_from_label(self, label, x):
        """Set row from label."""
        self.mat[self.rows_labels_ix_mapping[label]] = x

    def delete_row_from_label(self, label):
        """Set row from label."""
        self.mat = np.delete(self.mat, self.rows_labels_ix_mapping[label], 0)
        self.rows_labels.remove(label)
        self.update_ix_mappings()

    def get_col_from_label(self, label):
        """Get col from labels."""
        return self.mat[:, self.cols_labels_ix_mapping[label]]

    def delete_col_from_label(self, label):
        """Set col from label."""
        self.mat = np.delete(self.mat, self.cols_labels_ix_mapping[:, self.cols_labels_ix_mapping[label]], 1)
        self.cols_labels.remove(label)
        self.update_ix_mappings()

    def set_cell_from_labels(self, row_label, col_label, x):
        """Set cell from labels."""
        self.mat[self.rows_labels_ix_mapping[row_label], self.cols_labels_ix_mapping[col_label]] = x

    def get_cell_from_labels(self, row_label, col_label):
        """Get cell from labels."""
        return self.mat[self.rows_labels_ix_mapping[row_label], self.cols_labels_ix_mapping[col_label]]

    """Methods"""

    """Binds"""

    def row_bind(self, rows=None, rows_labels=None, matrix=None):
        """Return a copy of Matrix Object."""
        if matrix:
            rows = matrix.mat
            rows_labels = matrix.rows_labels

        if list(rows):
            self.mat = np.concatenate((self.mat, np.array(rows)), axis=0)
            self.rows_labels += rows_labels
            self.validate_labels_and_update_ix_mappings()
        else:
            log.warning('No column given to concatenate to matrix.')

    def col_bind(self, cols=None, cols_labels=None, matrix=None):
        """Return a copy of Matrix Object."""
        if matrix:
            cols = matrix.mat
            cols_labels = matrix.cols_labels

        if list(cols):
            self.mat = np.concatenate((self.mat, np.array(cols)), axis=1)
            self.cols_labels += cols_labels
            self.validate_labels_and_update_ix_mappings()
        else:
            log.warning('No column given to concatenate to matrix.')

    """Match matrices"""

    def match_rows(self, reference_matrix):
        """Match method to set rows labels as reference matrix."""
        if self.quadratic:
            log.warning('Changing rows of a symmetric Matrix implies changing also columns.')
            return self.match_mat(reference_matrix, True)

        mat_match = self.__copy__()
        mat_match.rows_labels = reference_matrix.rows_labels

        for row_label in reference_matrix.rows_labels:
            mat_match.mat[reference_matrix.rows_labels_ix_mapping[row_label]] = self.get_row_from_label(row_label)

        mat_match.validate_labels_and_update_ix_mappings()

        return mat_match

    def match_cols(self, reference_matrix):
        """Match method to set cols labels as reference matrix."""
        if reference_matrix.cols_labels == reference_matrix.cols_labels:
            return self

        if self.quadratic:
            log.warning('Changing columns of a symmetric Matrix implies changing also rows.')
            return self.match_mat(reference_matrix, True)

        mat_match = self.__copy__()
        mat_match.cols_labels = reference_matrix.cols_labels

        for col_label in reference_matrix.cols_labels:
            mat_match.mat[reference_matrix.cols_labels_ix_mapping[col_label]] = self.get_col_from_label(col_label)

        mat_match.validate_labels_and_update_ix_mappings()

        return mat_match

    def match_mat(self, reference_matrix, match_quadratic=None):
        """Match method to set axis labels as reference matrix."""
        if reference_matrix.cols_labels == self.cols_labels and reference_matrix.rows_labels == self.rows_labels:
            return self

        mat_match = self.__copy__()
        mat_match.rows_labels = reference_matrix.rows_labels

        if match_quadratic is None:
            match_quadratic = reference_matrix.quadratic

        if not match_quadratic:
            mat_match.cols_labels = reference_matrix.cols_labels
        else:
            Warning('Matching quadratic matrix: Same columns and row labels.')

        for score, row_label, col_label in iter(reference_matrix):
            mat_match.mat[reference_matrix.rows_labels_ix_mapping[row_label],
                          reference_matrix.cols_labels_ix_mapping[col_label]] \
                = self.get_cell_from_labels(row_label, col_label)

        mat_match.validate_labels_and_update_ix_mappings()

        return mat_match

    def match_missing_rows(self, reference_labels, missing_fill=0):
        """Match method to set missing rows labels from reference labels with the missing_fill value."""
        if reference_labels == self.rows_labels:
            return self

        missing_labels = set(reference_labels) - set(self.rows_labels)

        mat_match = self.__copy__()

        mat_match.rows_labels += list(missing_labels)

        missing_values = np.full((len(missing_labels), len(self.cols_labels)), missing_fill)

        mat_match.mat = np.concatenate((mat_match.mat, missing_values), axis=0)

        mat_match.validate_labels_and_update_ix_mappings()

        return mat_match

    def match_delete_rows(self, reference_labels):
        """Match method to set missing rows labels from reference labels with the missing_fill value."""
        if reference_labels == self.rows_labels:
            return self

        mat_match = self.__copy__()

        over_labels = set(mat_match.rows_labels) - set(reference_labels)

        for label in over_labels:
            mat_match.delete_row_from_label(label)

        mat_match.validate_labels_and_update_ix_mappings()

        return mat_match

    def match_missing_cols(self, reference_labels, missing_fill):
        """Match method to set missing cols labels from reference labels with the missing_fill value."""
        if reference_labels == self.cols_labels:
            return self

        mat_match = self.__copy__()

        missing_labels = set(reference_labels) - set(self.cols_labels)

        mat_match.cols_labels.append(missing_labels)

        missing_values = np.array([len(self.rows_labels), len(reference_labels.cols_labels)])
        missing_values.fill(missing_fill)

        mat_match.mat = np.concatenate(mat_match.mat, missing_values, axis=1)

        mat_match.validate_labels_and_update_ix_mappings()

        return mat_match

    """Order"""

    def order_rows(self, reverse=True, col_ref_idx=None):
        """Order matrix rows by cell values."""
        # Get the row index-cell value mapping.
        mapping = self.rows_idx_scores_mapping

        if len(self.mat[0]) != 1:
            if isinstance(col_ref_idx, int):
                mapping = {k: v[col_ref_idx] for k, v in mapping.items()}

            else:
                raise ValueError('Please use integers as indexes')

        # Get a list of index ordered by row values.
        idx_order = [k for k in sorted(mapping, key=mapping.get, reverse=reverse)]

        # Get a copy of the matrix object for not infer accessing values and to return.
        ordered_mat = self.__copy__()

        # Set the matrix's cells and row labels order according the previous ordered index list.
        for i, idx in enumerate(idx_order):
            ordered_mat.mat[i] = self.mat[idx]
            ordered_mat.rows_labels[i] = self.rows_labels[idx]

        return ordered_mat

    """Import"""

    def from_csv(self, csv_path):
        """Import matrix from csv file using the headers as a Matrix class."""
        m = np.genfromtxt(csv_path, dtype=None, delimiter=',')
        return Matrix(
            mat=np.array(
                [
                    [float(x)
                     for x in a[1:]]
                    for a in m[1:]
                ]),
            rows_labels=list(m[1:, 0]),
            cols_labels=list(m[0, 1:]),
            name=str(os.path.basename(csv_path).replace('.csv', ''))
        )

    """Export"""

    def to_dict(self, ordered=True):
        """Export/convert matrix as a dictionary data structure."""
        if ordered:
            mat = self.order_rows()
        else:
            mat = self

        # Construct dict first assigning the headers of rows_labels
        d = {'rows_labels': mat.rows_labels}
        for col_label in mat.cols_labels:
            d[col_label] = mat.get_col_from_label(col_label)

        return d

    def to_csv(self, path, file_name='_export.csv', index=False, ordered=True):
        """Export matrix to csv file using the headers (row_labels, cols_labels) of the Matrix class."""
        # Generate dataframe
        df = pd.DataFrame(data=self.to_dict(ordered))

        df.to_csv(os.path.join(path, self.name, file_name), index=index)


class LaplacianMatrix(Matrix):
    """Laplacian matrix class."""

    def __init__(self, graph, normalized=False, name=''):
        """Initialize laplacian."""
        l_mat = get_laplacian(graph, normalized)

        Matrix.__init__(self, mat=l_mat, quadratic=True, name=name, graph=graph)

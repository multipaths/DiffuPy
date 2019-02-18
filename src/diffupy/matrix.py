# -*- coding: utf-8 -*-

"""Main Matrix Class."""

import logging
import numpy as np
import os

from .miscellaneous import get_label_ix_mapping, get_label_list_graph, get_laplacian

log = logging.getLogger(__name__)


class Matrix:
    """Matrix class."""

    def __init__(self, mat, rows_labels=None, cols_labels=None, dupl=False, name='', graph=None, **kwargs):
        """Initialize matrix."""

        self._rows_labels = rows_labels

        if not dupl:
            self._cols_labels = cols_labels

        self._name = name
        self._dupl = dupl
        self._mat = np.array(mat)

        if graph:
            self._rows_labels= list(get_label_list_graph(graph, 'name'))

        self.set_mappings_validate_labels()

    def __str__(self):
        return f"matrix {self.name} \n {self.mat} \n row labels: {self.rows_labels} " \
               f"\n column labels: \n {self.cols_labels} \n : "

    """Iterator"""

    def __iter__(self, get_labels=True, get_indices=False):
        self.i = -1
        self.j = 0
        self.get_indices = get_indices
        self.get_labels = get_labels

        return self

    def __next__(self):
        if self.j >= len(self.rows_labels) - 1 and self.i >= len(self.cols_labels) - 1:
            raise StopIteration

        if self.i >= len(self.cols_labels) - 1:
            self.i = 0
            self.j += 1
        else:
            self.i += 1

        nxt = tuple()
        if len(self.rows_labels) == 1:
            nxt += (self.mat[self.i],)
        else:
            nxt += (self.mat[self.j][self.i],)

        if self.get_indices:
            nxt += (self.i, self.j,)

        if self.get_labels:
            nxt += (self.cols_labels[self.i], self.rows_labels[self.j],)

        return nxt

    """Copy"""

    def __copy__(self):
        """Return a copy of Matrix Object."""
        return Matrix(self.mat, rows_labels=self.rows_labels, cols_labels=self.cols_labels, dupl=self._dupl,
                      name=self.name)

    """Getters and Setters"""
    def set_mappings_validate_labels(self):

        if list(self.rows_labels):
            self._rows_labels_ix_mapping, self._rows_labels = get_label_ix_mapping(self.rows_labels)
        elif self.dupl and not list(self.cols_labels):
            log.warning(
                'Rows labels empty, also columns (neither cols labels given) will be empty since duplicate labels is true.')
        elif not self.dupl:
            log.warning('Rows labels empty.')

        if list(self.cols_labels):
            self.cols_labels_ix_mapping, self.cols_labels = get_label_ix_mapping(self.cols_labels)
            if self.dupl:
                log.warning('Columns labels are assigned to rows since duplicate labels is true.')
        elif not self.dupl:
            log.warning('Cols labels empty.')

    # Raw matrix (numpy array)
    @property
    def mat(self):
        return self._mat

    @mat.setter
    def mat(self, mat):
        self._mat = mat

    # Name
    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def dupl(self):
        return self._dupl

    @dupl.setter
    def dupl(self, dupl):
        self._dupl = dupl

    # Rows labels
    @property
    def rows_labels(self):
        return self._rows_labels

    @rows_labels.setter
    def rows_labels(self, rows_labels):
        self._rows_labels = list(rows_labels)

    # Columns labels
    @property
    def cols_labels(self):
        if self._dupl:
            return self._rows_labels

        return self._cols_labels

    @cols_labels.setter
    def cols_labels(self, cols_labels):
        if self._dupl:
            self._rows_labels = list(cols_labels)
        else:
            self._cols_labels = list(cols_labels)

    # Rows mapping
    @property
    def rows_labels_ix_mapping(self):
        return self._rows_labels_ix_mapping

    @rows_labels_ix_mapping.setter
    def rows_labels_ix_mapping(self, rows_labels_ix_mapping):
        self._rows_labels_ix_mapping = rows_labels_ix_mapping

    # Columns mapping
    @property
    def cols_labels_ix_mapping(self):
        if self._dupl:
            return self._rows_labels_ix_mapping

        return self._cols_labels_ix_mapping

    @cols_labels_ix_mapping.setter
    def cols_labels_ix_mapping(self, cols_labels_ix_mapping):
        if self._dupl:
            self._rows_labels_ix_mapping = cols_labels_ix_mapping
        else:
            self._cols_labels_ix_mapping = cols_labels_ix_mapping

    # From labels
    def set_row_from_label(self, label, x):
        self.mat[self.rows_labels_ix_mapping[label]] = x

    def get_row_from_label(self, label):
        return self.mat[self.rows_labels_ix_mapping[label]]

    def set_col_from_label(self, label, x):
        self.mat[:, self.cols_labels_ix_mapping[label]] = x

    def get_col_from_label(self, label):
        return self.mat[:, self.cols_labels_ix_mapping[label]]

    def set_from_labels(self, col_label, row_label, x):
        self.mat[self.rows_labels_ix_mapping[row_label], self.cols_labels_ix_mapping[col_label]] = x

    def get_from_labels(self, col_label, row_label):
        return self.mat[self.rows_labels_ix_mapping[row_label], self.cols_labels_ix_mapping[col_label]]

    # TODO: este nombre es un poco confuso no?

    """Methods"""

    """Binds"""

    def row_bind(self, rows=None, rows_labels=None, matrix=None):
        """Return a copy of Matrix Object."""
        if matrix:
            rows = matrix.mat
            rows_labels = rows_labels.rows_labels
        self.mat += rows
        self.rows_labels += rows_labels

    def col_bind(self, cols=None, cols_labels=None, matrix=None):
        """Return a copy of Matrix Object."""

        if matrix:
            cols = matrix.mat
            cols_labels = cols_labels.cols_labels
        self.mat[:, :-1] += cols
        self.cols_labels += cols_labels

    """Match matrices"""

    def match_rows(self, matrix_to_match):
        """Return a copy of Matrix Object."""
        if self.dupl:
            log.warning('Changing rows of a symmetric Matrix implies changing also columns.')
            return self.match_mat(matrix_to_match, True)

        mat_match = self.__copy__()
        mat_match.rows_labels = matrix_to_match.rows_labels

        for row_label in mat_match.rows_labels:
            mat_match.set_row_from_label(row_label, self.get_row_from_label(row_label))

        return mat_match

    def match_cols(self, matrix_to_match):
        if matrix_to_match.cols_labels == matrix_to_match.cols_labels:
            return self

        if self.dupl:
            log.warning('Changing columns of a symmetric Matrix implies changing also rows.')
            return self.match_mat(matrix_to_match, True)

        mat_match = self.__copy__()
        mat_match.cols_labels = matrix_to_match.cols_labels

        for row_label in mat_match.rows_labels:
            mat_match.set_col_from_label(row_label, self.get_row_from_label(row_label))

        return mat_match

    def match_mat(self, matrix_to_match, match_dupl=None):
        if matrix_to_match.cols_labels == matrix_to_match.cols_labels and matrix_to_match.rows_labels == matrix_to_match.rows_labels:
            return self

        mat_match = self.__copy__()
        mat_match.rows_labels = matrix_to_match.rows_labels

        if match_dupl is None:
            match_dupl = matrix_to_match.dupl
            Warning('Matrix to match symetric.')

        if not match_dupl:
            mat_match.cols_labels = matrix_to_match.cols_labels
        else:
            mat_match.dupl = True

        for score, col_label, row_label in iter(self):
            mat_match.set_from_labels(col_label, row_label, self.get_from_labels(col_label, row_label))

        return mat_match

    """Import"""

    def from_csv(path):
        """Import matrix from csv file using the headers as a Matrix class."""
        m = np.genfromtxt(path, dtype=None, delimiter=',')
        return Matrix(
            mat=np.array(
                [
                    [float(x)
                     for x in a[1:]]
                    for a in m[1:]
                ]),
            rows_labels=m[1:, 0],
            cols_labels=m[0, 1:],
            name=str(os.path.basename(path).replace('.csv', ''))
        )


# TODO: Poner que es matriz simetrica
class LaplacianMatrix(Matrix):
    def __init__(self, graph, normalized=False, name=''):
        l_mat = get_laplacian(graph, normalized)
        Matrix.__init__(self, l_mat, dupl=True, name=name, graph=graph)

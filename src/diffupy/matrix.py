# -*- coding: utf-8 -*-

"""Main Matrix Class."""

import os

import numpy as np

from .miscellaneous import get_label_ix_mapping, get_label_list_graph, get_laplacian


class Matrix:
    """Matrix class."""

    # TODO: why kw?
    def __init__(self, mat, rows_labels=None, cols_labels=None, dupl=False, name='', graph=None, **kw):
        """Initialize matrix."""

        self._rows_labels = rows_labels

        self._cols_labels = cols_labels

        self._name = name
        self._dupl = dupl
        self._mat = np.array(mat)

        if graph:
            self._rows_labels = get_label_list_graph(graph, 'name')

        self.rows_labels_ix_mapping = get_label_ix_mapping(self.rows_labels)
        self.cols_labels_ix_mapping = get_label_ix_mapping(self.rows_labels)

    def __str__(self):
        return f"matrix {self.name} \n {self.mat} \n row labels: {self.rows_labels} " \
               f"\n column labels: \n {self.cols_labels} \n : "


    """Iterator"""

    def __iter__(self):
        self.i = -1
        self.j = 0
        return self

    def __next__(self):
        if self.j >= len(self.rows_labels)-1 and self.i >= len(self.cols_labels)-1:
            raise StopIteration

        if self.i >= len(self.cols_labels)-1:
            self.i = 0
            self.j += 1
        else:
            self.i += 1

        if len(self.rows_labels) == 1: return self.mat[self.i], self.cols_labels[self.i], self.rows_labels[self.j]
        return self.mat[self.j][self.i], self.cols_labels[self.i], self.rows_labels[self.j]

    """Copy"""

    def __copy__(self):
        if self._dupl:
            return Matrix(self.mat, rows_labels = self.rows_labels, dupl = True, name = self.name)
        else:
            return Matrix(self.mat, rows_labels = self.rows_labels, cols_labels = self.cols_labels, name = self.name)

    """Getters and Setters"""

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
        else:
            return self._cols_labels

    @cols_labels.setter
    def cols_labels(self, cols_labels):
        if self._dupl:
            self._rows_labels = list(cols_labels)
        else:
            self._cols_labels = list(cols_labels)

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

    """Match matrices"""

    def match_rows(self, matrix_to_match):
        if self.dupl:
            Warning('Changing rows of a symetric Matrix.')
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
            Warning('Changing columns of a symetric Matrix.')
            return self.match_mat(matrix_to_match, True)

        mat_match = self.__copy__()
        mat_match.cols_labels = matrix_to_match.cols_labels

        for row_label in mat_match.rows_labels:
            mat_match.set_col_from_label(row_label, self.get_row_from_label(row_label))

        return mat_match


    def match_mat(self, matrix_to_match, match_dupl = None):
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

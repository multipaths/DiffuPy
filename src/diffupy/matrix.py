# -*- coding: utf-8 -*-

"""Main Matrix Class."""

import os

import numpy as np

from .miscellaneous import get_label_id_mapping, get_label_list_graph, get_laplacian


class Matrix:
    """Matrix class."""

    # TODO: why kw?
    def __init__(self, mat, rows_labels=None, cols_labels=None, name='', graph=None, dupl=False, **kw):
        """Initialize matrix."""

        self._rows_labels = rows_labels

        self._cols_labels = cols_labels

        self._name = name
        self._dupl = dupl
        self._mat = np.array(mat)

        if graph:
            self._rows_labels = get_label_list_graph(graph, 'name')

        self.label_id_mapping = get_label_id_mapping(self.rows_labels, self.cols_labels)

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
        i_row = self.label_id_mapping[label]
        if isinstance(i_row, dict):
            i_row = i_row[label].values()[0][0]
        self.mat[i_row] = x

    def get_row_from_label(self, label):
        i_row = self.label_id_mapping[label]
        if isinstance(i_row, dict):
            i_row = i_row[label].values()[0][0]
        return self.mat[i_row]

    def set_col_from_label(self, label, x):
        i_col = self.label_id_mapping[label]
        if isinstance(i_col, dict):
            i_col = i_col[label].values()[0][0]
        self.mat[:, i_col] = x

    def get_col_from_label(self, label, x):
        i_col = self.label_id_mapping[label]
        if isinstance(i_col, dict):
            i_col = i_col[label].values()[0][0]
        return self.mat[:, i_col]

    def set_from_labels(self, row_label, col_label, x):
        self.mat[self.label_id_mapping[row_label][col_label]] = x

    def get_from_labels(self, row_label, col_label):
        return self.mat[self.label_id_mapping[row_label][col_label]]

    # TODO: este nombre es un poco confuso no?
    def match_matrix(self):
        return self.rows_labels

    """Methods"""

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
            name=str(os.path.basename(path).split('.csv'))
        )


# TODO: Poner que es matriz simetrica
class LaplacianMatrix(Matrix):
    def __init__(self, graph, normalized=False, name=''):
        l_mat = get_laplacian(graph, normalized)
        Matrix.__init__(self, l_mat, name=name, dupl=True, graph=graph)

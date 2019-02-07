# -*- coding: utf-8 -*-

"""Miscellaneous utils of the package."""

import numpy as np

from .miscellaneous import get_label_id_mapping, get_label_list_graph


class Matrix:
    """Matrix class."""

    def __init__(self, mat, name='', rows_labels=[], cols_labels=[], dupl=False, graph=None, **kw):
        """Initialize matrix."""
        self._name = name
        self._mat = np.array(mat)
        self._rows_labels = rows_labels
        self._cols_labels = cols_labels

        # TODO: si no hay graph es dupl? porque te puedes ahorrar una variable
        if graph:
            self._rows_labels = get_label_list_graph(graph)

        if dupl:
            self._cols_labels = self._rows_labels

        # TODO: check cols_labels argument
        self.label_id_mapping = get_label_id_mapping(mat, rows_labels, cols_labels)

    def __str__(self):
        return f"matrix {self.name} \n {self.mat} \n row labels: {self.rows_labels} " \
               f"\n column labels: \n {self.cols_labels} \n : "

    # Getters - setters
    # Raw matrix (numpy array)
    @property
    def mat(self):
        return self._mat

    @mat.setter
    def mat(self, mat):
        self._mat = mat

    # Matrix title
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
        return self._cols_labels

    @cols_labels.setter
    def cols_labels(self, cols_labels):
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

    # tODO: este nombre es un poco confuso no?
    def match_matrix(self):
        return self.rows_labels

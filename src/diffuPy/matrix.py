# -*- coding: utf-8 -*-

"""Main Matrix Class."""

import logging
import os
from collections import defaultdict

import numpy as np

from .miscellaneous import get_label_ix_mapping, get_label_list_graph, get_laplacian

log = logging.getLogger(__name__)

class Matrix:
    """Matrix class."""

    def __init__(self, mat=None, rows_labels=None, cols_labels=None, quadratic=False, name='', graph=None, init=None,
                 no_duplicates = True, **kwargs):
        """Initialize matrix."""

        if rows_labels:
            self._rows_labels = list(rows_labels)

        if graph:
            self._rows_labels = get_label_list_graph(graph, 'name')

        if not quadratic:
            self._cols_labels = list(cols_labels)

        self._name = name
        self._quadratic = quadratic

        if init and self.rows_labels and self.cols_labels:
            mat = np.full((len(self.rows_labels), len(self.cols_labels)), init)
        elif not list(mat):
            raise ValueError('An input matrix or initialization should be provided.')

        self._mat = np.array(mat)

        self.get_labels = True
        self.get_indices = False

        self.no_duplicates = no_duplicates

        self.set_mappings_and_validate_labels()

    def __str__(self):
        return f"\nmatrix {self.name} \n  {self.mat} \n row labels: \n  {self.rows_labels} " \
            f"\n column labels: \n  {self.cols_labels} \n "

    """Iterator"""

    def __iter__(self, **kargs):
        self.i = -1
        self.j = 0

        if 'get_indices' in kargs:
            self.get_indices = kargs['get_indices']
        if 'get_labels' in kargs:
            self.get_labels = kargs['get_labels']

        return self

    def __next__(self):
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

    """Copy"""

    def __copy__(self):
        """Return a copy of Matrix Object."""
        return Matrix(self.mat, rows_labels=self.rows_labels, cols_labels=self.cols_labels, quadratic=self._quadratic,
                      name=self.name)

    """Validators """
    def validate_duplicates(self):
        row_labels = []
        col_lables = []
        rep_col = defaultdict(lambda : dict())

        mat = np.empty((len(set(self.rows_labels)),len(set(self.cols_labels))))

        for value, row_index, col_index, row_label, col_label in self.__iter__(get_indices=True, get_labels=True):

            if row_label in row_labels:
                mat[row_labels.index(row_label)] = np.sum([self.mat[row_labels.index(row_label)], self.mat[row_index]])

                if self.quadratic:
                    for col_label, cols in rep_col.items():
                        for col_index, col in cols.items():
                            rep_col[col_label][col_index] = np.delete(col, row_index)

            if self.quadratic:
                if col_label in col_lables:
                    rep_col[col_label][col_index] = self.mat[:,col_index]
                else:
                    col_lables.append(col_label)


            else:
                np.append(mat, self.mat[row_index])
                row_labels.append(row_label)

        for col_label, cols in rep_col.items():
            for col_index, col in cols.items():
                mat[:, col_lables.index(col_label)] = np.sum([mat[:, col_lables.index(col_label)], self.mat[:, col_index]])
            for col_index, col in cols.items():
                np.delete(mat, col_index, 1)



        self.mat = mat
        self.rows_labels = row_labels

        if self.quadratic:
            self.col_lables = col_lables

    def set_mappings_and_validate_labels(self):
        # if self.no_duplicates and (set(self.rows_labels) != self.rows_labels or set(self.cols_labels) != self.cols_labels):
            # self.validate_duplicates()

        if self.rows_labels:
            self._rows_labels_ix_mapping, self._rows_labels = get_label_ix_mapping(self.rows_labels)
        elif self.quadratic and not list(self.cols_labels):
            log.warning(
                'Rows labels empty, also columns (neither cols labels given) will be empty since duplicate labels is true.')
        elif not self.quadratic:
            log.warning('Rows labels empty.')

        if list(self.cols_labels):
            self.cols_labels_ix_mapping, self.cols_labels = get_label_ix_mapping(self.cols_labels)
            if self.quadratic:
                log.warning('Columns labels are assigned to rows since duplicate labels is true.')
        elif not self.quadratic:
            log.warning('Cols labels empty.')

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
    def quadratic(self):
        return self._quadratic

    @quadratic.setter
    def quadratic(self, quadratic):
        self._quadratic = quadratic

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
        if self._quadratic:
            return self._rows_labels

        return self._cols_labels

    @cols_labels.setter
    def cols_labels(self, cols_labels):
        if self._quadratic:
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
        if self._quadratic:
            return self._rows_labels_ix_mapping

        return self._cols_labels_ix_mapping

    @cols_labels_ix_mapping.setter
    def cols_labels_ix_mapping(self, cols_labels_ix_mapping):
        if self._quadratic:
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

    def set_from_labels(self, row_label, col_label, x):
        self.mat[self.rows_labels_ix_mapping[row_label], self.cols_labels_ix_mapping[col_label]] = x

    def get_from_labels(self, row_label, col_label, ):
        return self.mat[self.rows_labels_ix_mapping[row_label], self.cols_labels_ix_mapping[col_label]]

    # TODO: este nombre es un poco confuso no?

    """Methods"""

    """Binds"""

    def row_bind(self, rows=None, rows_labels=None, matrix=None):
        """Return a copy of Matrix Object."""

        if matrix:
            rows = matrix.mat
            rows_labels = rows_labels.rows_labels

        if list(rows):
            self.mat = np.concatenate((self.mat, np.array(rows)), axis=0)
            self.rows_labels += rows_labels
            self.set_mappings_and_validate_labels()
        else:
            log.warning('No column given to concatenate to matrix.')

    def col_bind(self, cols=None, cols_labels=None, matrix=None):
        """Return a copy of Matrix Object."""

        if matrix:
            cols = matrix.mat
            cols_labels = cols_labels.cols_labels

        if list(cols):
            self.mat = np.concatenate(self.mat, cols, axis=1)
            self.cols_labels += cols_labels
            self.set_mappings_and_validate_labels()
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

        mat_match.set_mappings_and_validate_labels()

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

        mat_match.set_mappings_and_validate_labels()

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
            mat_match.mat[reference_matrix.rows_labels_ix_mapping[row_label], \
                          reference_matrix.cols_labels_ix_mapping[col_label]] \
                = self.get_from_labels(row_label, col_label)

        mat_match.set_mappings_and_validate_labels()

        return mat_match

    def match_missing_rows(self, reference_labels, missing_fill):
        """Match method to set missing rows labels from reference labels with the missing_fill value."""

        if reference_labels == self.rows_labels:
            return self


        missing_labels = set(reference_labels) - set(self.rows_labels)

        mat_match = self.__copy__()

        mat_match.rows_labels += list(missing_labels)

        missing_values = np.full((len(missing_labels), len(self.cols_labels)), missing_fill)

        mat_match.mat = np.concatenate((mat_match.mat, missing_values), axis=0)

        mat_match.set_mappings_and_validate_labels()


        return mat_match

    def match_missing_cols(self, reference_labels, missing_fill):
        """Match method to set missing cols labels from reference labels with the missing_fill value."""

        if reference_labels == self.cols_labels:
            return self

        mat_match = self.__copy__()

        missing_labels = set(reference_labels) - set(self.cols_labels)

        mat_match.cols_labels.append(missing_labels)

        missing_values = np.array([len(missing_labels), len(reference_labels.cols_labels)])
        missing_values.fill(missing_fill)

        mat_match.mat = np.concatenate(mat_match.mat, missing_values, axis=1)

        mat_match.set_mappings_and_validate_labels()

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

        Matrix.__init__(self, mat=l_mat, quadratic=True, name=name, graph=graph)

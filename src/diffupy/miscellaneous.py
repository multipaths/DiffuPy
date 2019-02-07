import networkx as nx
import numpy as np
import sys

# General functions
# Labels mapping
def get_laplacian(G, normalized=False):
    if nx.is_directed(G):
        sys.exit('Graph must be undirected')

    if not normalized:
        L = nx.laplacian_matrix(G).toarray()
    else:
        L = nx.normalized_laplacian_matrix(G).toarray()

    return L

def get_label_list_graph(graph):
    return [v for k, v in nx.get_node_attributes(graph, 'name')]


def get_label_id_mapping(labels_row, labels_col = None):
    if not labels_col:
        return {label: i for i, label in enumerate(labels_row)}
    else:
        return {label_row:{label_col: (j, i) for j, label_col in enumerate(labels_col)} for i, label_row in enumerate(labels_row)}


# Matrix class
class Matrix:
    def __init__(self, mat, name='', rows_labels=[], cols_labels=[], dupl=False, graph=None, **kw):
        self._name = name
        self._mat = np.array(mat)
        self._rows_labels = rows_labels
        self._cols_labels = cols_labels

        if graph:
            self._rows_labels = get_label_list_graph(graph)

        if dupl:
            self._cols_labels = self._rows_labels

        label_id_mapping = get_label_id_mapping(mat, rows_labels, cols_labels)

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

    def match_matrix(self):
        return self.rows_labels

    def __str__(self):
        return "matrix %s \n %s \n row_labels \n : %s \n column_labels \n : %s" % (
        self.name, self.mat, self.rows_labels, self.cols_labels)


class LaplacianMatrix(Matrix):
    def __init__(self, graph, normalized=False, name=''):
        l_mat = get_laplacian(graph, normalized)
        Matrix.__init__(self, l_mat, name=name, dupl=True, graph=graph)
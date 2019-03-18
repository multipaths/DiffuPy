import logging

import numpy as np

from diffuPy.diffuse_raw import diffuse_raw
from diffuPy.miscellaneous import get_label_list_graph
from diffuPy.validate_inputs import _validate_scores

log = logging.getLogger(__name__)


def diffuse(scores, method, graph=None, **kwargs):
    # sanity checks
    _validate_scores(scores)

    # Check if we have a graph or a kernel
    if graph:
        format_network = "graph"
    else:
        if not "K" in kwargs:
            raise ValueError("Neither a graph 'graph' or a kernel 'K' has been provided.")
        format_network = "kernel"

    # TODO: que pasa si es kernel y method == 'raw'? va a entrar aqui y va a petar todo (yo meteria esta parte que
    #  trabaja con el graph dentro del if de antes
    # Diffuse raw
    if method == "raw":
        return diffuse_raw(graph=graph, scores=scores, **kwargs)

    # z scores
    elif method == "z":
        return diffuse_raw(graph, scores, z=True, **kwargs)

    elif method == "ml":
        for score, i, j in scores.__iter__(get_labels=False, get_indices=True):
            if score not in [0, 1]:
                raise ValueError("'graph' cannot have NA as node names")
            if score == 0:
                scores.mat[i, j] = -1

        return diffuse_raw(graph, scores, **kwargs)

    elif method == "gm":
        for score, i, j in scores.__iter__(get_labels=False, get_indices=True):
            if score not in [0, 1]:
                raise ValueError("Input scores must be binary.")
                # Have to match rownames with background
                # If the kernel is provided...

        if format_network == "graph":
            names_ordered = get_label_list_graph(graph, 'name')
        elif format_network == "kernel":
            names_ordered = kwargs['K'].rows_labels

        # If the graph is defined...
        ids_nobkgd = set(names_ordered) - set(scores.rows_labels)

        n_tot = len(names_ordered)
        n_bkgd = scores.mat.shape[0]

        # normalisation has to be performed
        # for each column, as it depends
        # on the number of positives and negatives...
        # n_pos and n_neg are vectors counting the number of
        # positives and negatives in each column
        n_pos = np.sum(scores.mat, axis=0)
        n_neg = n_bkgd - n_pos

        # biases
        p = (n_pos - n_neg) / n_tot

        for score, i, j in scores.__iter__(get_labels=False, get_indices=True):
            if score == 0:
                scores.mat[i, j] = -1

        # add biases (each column has its bias)
        scores.row_bind(
            np.transpose(np.array(
                [np.repeat(
                    score,
                    n_tot - n_bkgd
                )
                    for score in p
                ])
            ),
            ids_nobkgd
        )

        return diffuse_raw(graph, scores, **kwargs)

    else:
        raise ValueError("")

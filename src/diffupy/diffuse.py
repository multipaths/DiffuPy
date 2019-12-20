# -*- coding: utf-8 -*-

"""Diffusion methods."""

import copy
import logging

import numpy as np

from .diffuse_raw import diffuse_raw
from .utils import get_label_list_graph
from .validate_input import _validate_scores

log = logging.getLogger(__name__)

def diffuse(input_scores, method, graph = None, **kwargs):
    """
    Generalized function to treat different methods of score diffusion on a network / modeling heat diffusion.

    It takes a network (as a graph [graph] or as a kernel transformation [K in kwargs]
    as an optional argument / but mandatory if graph not provided (managed programaticaly).

    Diffusion methods procedures provided in this package differ on:
        (a) How to distinguish positives, negatives and unlabelled examples.
        (b) Their statistical normalisation.

    #  TODO: Acordar de especificar mejor el diferente tratamiento de los inputs, checkear gramma too.
    Input scores can be specified in three formats: a single set of scores to smooth can be represented as.
        (1) a named numeric vector, whereas if several of these vectors that share the node names need to be smoothed,
            they can be provided as
        (2) a column-wise matrix. However, if the unlabelled entities are not the same from one case to another,
        (3) a named list of such score matrices can be passed to this function. The input format will be kept in the output.
    If the input labels are not quantitative, i.e. positive(1), negative(0) and possibly unlabelled, all the scores
    raw, gm, ml, z, mc, ber_s, ber_p can be used.

    Methods [method attribute to choose one node]:
        - Methods without statistical normalisation:

            {raw}:  positive nodes {y_raw[i] = 1} introduce unitary flow to
                    the network, whereas either negative and unlabelled
                    nodes introduce null diffusion. {y_raw[j] = 0}
                    [Vandin, 2011]

            {ml}:   same as raw, but negative nodes introduce a negative unit of flow.
                    therefore not equivalent to unlabelled nodes.
                    [Zoidi, 2015]

            {gm}:   same as ml, but the unlabelled nodes are assigned
                    a (generally non-null) bias term based on the total
                    number of positives, negatives and unlabelled nodes
                    [Mostafavi, 2008]

            {ber_s}: a quantification of the relative change in the node score before
                     and after the network smoothing.

        - Methods with statistical normalisation:
            {z}: a parametric alternative to the raw score of node is subtracted its mean
                 value and divided by its standard deviation. Differential trait of this package.
                 The statistical moments have a closed analytical form,
                 see the main vignette, and are inspired in [Harchaoui, 2013].

            {mc}: the score of node code {i} is based on its empirical p-value, computed by permuting the input {n.perm} times.
                  It is roughly the proportion of input permutations that led to a diffusion score as high or higher than the
                  original diffusion score.

            {ber_p}: as used in [Bersanelli, 2016], this score combines raw and mc, in order to take into
                     account both the magnitude of the \code{raw} scores and the effect of the network topology :
                     this is a quantification of the relative change in the node score before and after the network smoothing.

    Methods summary table:
     __ __  __ __   __ __   __ __  __ __  __ __  __ __  __ __ __ __ __ __ __ __ __ __ __ __
    | Scores  | y+  | y- | yn |  Normalized | Stochastic | Quantitative | Reference        |
     __ __  __ __   __ __   __ __  __ __  __ __  __ __  __ __ __ __ __ __ __ __ __ __ __ __
     __Unormalized__  __ __  __ __  __ __  __ __  __ __  __ __  __ __  __ __ __ __ __ __ __
    | raw     |  1  | 0  |  0 |      No     |     No     |     Yes      | Vandin (2010)    |
     __ __  __ __  __ __  __ __  __ __  __ __  __ __  __ __ __ __  __ __  __ __  __ __ __ _
    | ml      |  1  | -1 |  0 |      No     |     No     |     No       | Tsuda (2010)     |
     __ __  __ __  __ __  __ __  __ __  __ __  __ __  __ __ __ __  __ __  __ __  __ __ __ _
    | gm      |  1  | -1 |  k |      No     |     No     |     No       | Mostafavi (2008) |
      __ __  __ __  __ __  __ __  __ __  __ __  __ __  __ __ __ __  __ __  __ __  __ __ __
    | ber_s   |  1  | 0  |  0 |      No     |     No     |     Yes      | Bersanelli (2016)|
     __ __  __ __  __ __  __ __  __ __  __ __  __ __  __ __ __ __  __ __  __ __  __ __ __ _
     __Normalized  __ __  __ __  __ __  __ __  __ __  __ __ __ __  __ __  __ __  __ __ __ _
    | ber_p   |  1  | 0  | 0* |      Yes    |     Yes    |     Yes      | Bersanelli (2016)|
     __ __  __ __  __ __  __ __  __ __  __ __  __ __  __ __ __ __  __ __  __ __  __ __ __
    | mc      |  1  | 0  | 0* |      Yes    |     Yes    |     Yes      | Bersanelli (2016)|
     __ __  __ __  __ __  __ __  __ __  __ __  __ __  __ __ __ __  __ __  __ __  __ __ __
    | z       |  1  | 0  | 0* |      Yes    |     No     |     Yes      | Harchaoui (2013) |
     __ __  __ __  __ __  __ __  __ __  __ __  __ __  __ __ __ __  __ __  __ __  __ __ __ _

    :param input_scores: A vector input.
    :param method: One of the possible methods described previously.
                   Possible values ["raw", "ml", "gm", "ber_s", "ber_p", "mc", "z"]
    :param graph: A network in a graph format.
    :param kwargs: Optional arguments:
                    - K: a  kernel [matrix] transformation from a graph.
                    - Other arguments, that would differ depending the chosen method.
    :return: The diffused scores within the matrix transformation of the network, with the diffusion operation
             [K x input_vector] performed.

    """

    # sanity checks
    scores = copy.copy(input_scores)

    _validate_scores(scores)


    # Check if we have a graph or a kernel
    if graph:
        format_network = "graph"
    else:
        if not "K" in kwargs:
            raise ValueError("Neither a graph 'graph' or a kernel 'K' has been provided.")
        format_network = "kernel"

    #  TODO: que pasa si es kernel y method == 'raw'? va a entrar aqui y va a petar todo (yo meteria esta parte que
    #   trabaja con el graph dentro del if de antes
    # Diffuse raw
    if method == "raw":
        return diffuse_raw(graph=graph, scores=scores, **kwargs)

    # z scores
    elif method == "z":
        return diffuse_raw(graph, scores, z=True, **kwargs)

    elif method == "ml":
        for score, i, j in scores.__iter__(get_labels=False, get_indices=True):
            if score not in [-1, 0, 1]:
                raise ValueError("Input scores must be binary.")
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

import random
from collections import defaultdict

import networkx as nx
import numpy as np
from tqdm import tqdm

from diffupy.diffuse import diffuse
from diffupy.matrix import Matrix
from diffupy.utils import get_simplegraph_from_multigraph
from .input_mapping import generate_categoric_input_from_labels

from sklearn import metrics

def get_random_cv_split_input_and_validation(input, background_mat):
    randomized_input_labels = random.sample(population=list(input), k=int(len(input) / 2))
    validation_labels = list(set(input) - set(randomized_input_labels))

    return generate_categoric_input_from_labels(randomized_input_labels, 'randomized input', background_mat), \
           generate_categoric_input_from_labels(validation_labels, 'validation labels', background_mat)


def get_random_cv_inputs_from_subsets_same_diff_input(input_subsets, background_mat):
    input_labels = set()
    validation_mats_by_entity_type = defaultdict()

    for entity_type, input in input_subsets.items():
        randomized_input_labels = random.sample(population=list(input[0]), k=int(len(input[0]) / 2))
        input_labels.update(set(randomized_input_labels))

        validation_labels = list(set(input[0]) - set(randomized_input_labels))
        validation_mats_by_entity_type[entity_type] = generate_categoric_input_from_labels(validation_labels,
                                                                                           'Dataset 1 ' + str(
                                                                                               entity_type),
                                                                                           background_mat)

        # Match testing
        # if entity_type == 'metabolite':
        #     for score, i, j, row_label, col_label in validation_mats_by_entity_type[entity_type].__iter__(get_indices = True, get_labels = True):
        #         if score == 1:
        #             print(row_label)

    input_mat = generate_categoric_input_from_labels(input_labels, 'Dataset1', background_mat)

    # Match testing
    # total = set()
    # for score, i, j, row_label, col_label in input_mat.__iter__(get_indices = True, get_labels = True):
    #     if score == 1:
    #         total.add(row_label)
    #
    # print(len(input_labels))
    # print(len(total))

    return input_mat, validation_mats_by_entity_type


def get_metrics(validation_labels, scores):
    return metrics.roc_auc_score(validation_labels.mat, scores.mat), metrics.average_precision_score(
        validation_labels.mat, scores.mat)


def cross_validation_by_subset_same_diff_input(mapping_by_subsets, kernel, k=3, z=True):
    auroc_metrics = defaultdict(list)
    auprc_metrics = defaultdict(list)

    for i in tqdm(range(k)):
        input_mat, validation_inputs_by_subsets = get_random_cv_inputs_from_subsets_same_diff_input(mapping_by_subsets,
                                                                                                    kernel)

        scores = diffuse(input_mat, 'ml', K=kernel, z=z)

        for entity, validation_labels in validation_inputs_by_subsets.items():
            auroc, auprc = get_metrics(validation_labels, scores)
            auroc_metrics[entity].append(auroc)
            auprc_metrics[entity].append(auprc)

    return auroc_metrics, auprc_metrics


def generate_pagerank_baseline(graph, background_mat):
    graph = get_simplegraph_from_multigraph(graph)

    pagerank_scores = nx.pagerank(graph)

    return Matrix(mat=np.array(
                                list(pagerank_scores.values())).reshape(
                                    (len(list(pagerank_scores.values())),1)
                                ),
                    rows_labels=list(pagerank_scores.keys()),
                    cols_labels=['PageRank']
                ).match_missing_rows(background_mat.rows_labels, 0).match_rows(background_mat)


def generate_random_score_ranking(background_mat):
    return Matrix(mat = np.random.rand(len(background_mat.rows_labels)),
                   rows_labels = background_mat.rows_labels,
                   cols_labels = ['Radom_Baseline']
                )

def cross_validation_by_method(all_labels_mapping, graph, kernel, k=3):
    auroc_metrics = defaultdict(list)
    auprc_metrics = defaultdict(list)

    input_mat = generate_categoric_input_from_labels(all_labels_mapping, 'Dataset1', kernel)

    scores_page_rank = generate_pagerank_baseline(graph, kernel)


    for i in tqdm(range(k)):
        input_diff, validation_diff = get_random_cv_split_input_and_validation(
            all_labels_mapping, kernel
        )
        scores_z = diffuse(input_diff, 'ml', K=kernel, z=True)
        scores_raw = diffuse(input_diff, 'ml', K=kernel, z=False)

        method_validation_inputs = {
             'raw' : (validation_diff,
                      scores_raw
                    ),
             'z' : (validation_diff,
                    scores_z
                    ),
            'page_rank_baseline' : (validation_diff,
                                    scores_page_rank
                                    ),
            'random_baseline' : (validation_diff,
                                 generate_random_score_ranking(kernel)
                                 ),
            }

        for method, validation_set in method_validation_inputs.items():
            auroc, auprc = get_metrics(*validation_set)
            auroc_metrics[method].append(auroc)
            auprc_metrics[method].append(auprc)

    return auroc_metrics, auprc_metrics




from collections import defaultdict

import networkx as nx
import numpy as np
from tqdm import tqdm

from diffupy.diffuse_raw import diffuse_raw
from diffupy.matrix import Matrix
from diffupy.utils import get_simplegraph_from_multigraph, split_random_two_subsets, random_disjoint_intersection_three_subsets
from diffupy.input_mapping import generate_categoric_input_from_labels

from sklearn import metrics

import itertools

# Random cross validation

def get_random_cv_split_input_and_validation(input, background_mat):
    randomized_input_labels, validation_labels = split_random_two_subsets(input)

    return generate_categoric_input_from_labels(randomized_input_labels,
                                                'randomized input',
                                                background_mat), \
           generate_categoric_input_from_labels(validation_labels,
                                                'validation labels',
                                                background_mat)


def get_random_cv_inputs_from_subsets_same_diff_input(input_subsets, background_mat):
    input_labels = set()
    input_unlabeled = set()

    validation_mats_by_entity_type = defaultdict()

    for entity_type, input in input_subsets.items():
        randomized_input_labels, validation_labels = split_random_two_subsets(input[0])
        validation_mats_by_entity_type[entity_type] = generate_categoric_input_from_labels(validation_labels,
                                                                                           'Dataset 1 ' + str(entity_type),
                                                                                           background_mat
                                                                                           )
        input_unlabeled.update(set(validation_labels))
        input_labels.update(set(randomized_input_labels))
        # Match testing
        # if entity_type == 'metabolite':
        #     for score, i, j, row_label, col_label in validation_mats_by_entity_type[entity_type].__iter__(get_indices = True, get_labels = True):
        #         if score == 1:
        #             print(row_label)

    input_mat = generate_categoric_input_from_labels(input_labels, 'Dataset1', background_mat, input_unlabeled)
    # Match testing
    # total = set()
    # for score, i, j, row_label, col_label in input_mat.__iter__(get_indices = True, get_labels = True):
    #     if score == 1:
    #         total.add(row_label)
    #
    # print(len(input_labels))
    # print(len(total))

    return input_mat, validation_mats_by_entity_type

# Partial cross validation

def get_one_x_in_cv_inputs_from_subsets(input_subsets, background_mat, one_in ='Reactome'):

    input_labels= input_subsets.pop(one_in)
    validation_labels = set(itertools.chain.from_iterable(input_subsets.values()))


    return generate_categoric_input_from_labels(input_labels,
                                                 'two out input',
                                                 background_mat,
                                                 rows_unlabeled = validation_labels), \
           generate_categoric_input_from_labels(validation_labels,
                                                 'two out input',
                                                 background_mat
                                                )


def get_metrics(validation_labels, scores):
    return metrics.roc_auc_score(validation_labels.mat, scores.mat), metrics.average_precision_score(
        validation_labels.mat, scores.mat)




def cross_validation_by_subset_same_diff_input(mapping_by_subsets, kernel, k=3, z=True):
    auroc_metrics = defaultdict(list)
    auprc_metrics = defaultdict(list)

    for i in tqdm(range(k)):
        input_mat, validation_inputs_by_subsets = get_random_cv_inputs_from_subsets_same_diff_input(mapping_by_subsets,
                                                                                                    kernel)

        scores = diffuse_raw(graph = None, scores = input_mat, K=kernel, z=z)

        for entity, validation_labels in validation_inputs_by_subsets.items():
            auroc, auprc = get_metrics(validation_labels, scores)
            auroc_metrics[entity].append(auroc)
            auprc_metrics[entity].append(auprc)

    return auroc_metrics, auprc_metrics


def cross_validation_one_x_in(mapping_by_subsets, kernel, k=3, disjoint = False, z=True):
    auroc_metrics = defaultdict(list)
    auprc_metrics = defaultdict(list)

    for i in tqdm(range(k)):
        if disjoint:
            mapping_by_subsets = random_disjoint_intersection_three_subsets(mapping_by_subsets)
        else:
            mapping_by_subsets = {db : v[0] for db, v in mapping_by_subsets.items()}

        for subset_type in tqdm(mapping_by_subsets):
            input_mat, validation_labels = get_one_x_in_cv_inputs_from_subsets(dict(mapping_by_subsets),
                                                                               kernel,
                                                                               one_in = subset_type)

            scores = diffuse_raw(graph = None, scores = input_mat, K=kernel, z=z)

            auroc, auprc = get_metrics(validation_labels, scores)

            auroc_metrics[subset_type].append(auroc)
            auprc_metrics[subset_type].append(auprc)

    return auroc_metrics, auprc_metrics

# Method cross validation

def generate_pagerank_baseline(graph, background_mat):
    graph = get_simplegraph_from_multigraph(graph)

    pagerank_scores = nx.pagerank(graph)

    return Matrix(mat=np.array(
        list(pagerank_scores.values())).reshape(
        (len(list(pagerank_scores.values())), 1)
    ),
        rows_labels=list(pagerank_scores.keys()),
        cols_labels=['PageRank']
    ).match_missing_rows(background_mat.rows_labels, 0).match_rows(background_mat)


def generate_random_score_ranking(background_mat):
    return Matrix(mat=np.random.rand(len(background_mat.rows_labels)),
                  rows_labels=background_mat.rows_labels,
                  cols_labels=['Radom_Baseline']
                  )


def cross_validation_by_method(all_labels_mapping, graph, kernel, k=3):
    auroc_metrics = defaultdict(list)
    auprc_metrics = defaultdict(list)


    scores_page_rank = generate_pagerank_baseline(graph, kernel)

    for i in tqdm(range(k)):
        input_diff, validation_diff = get_random_cv_split_input_and_validation(
            all_labels_mapping, kernel
        )

        scores_z = diffuse_raw(graph = None, scores = input_diff, K=kernel, z=True)
        scores_raw = diffuse_raw(graph = None, scores = input_diff, K=kernel, z=False)

        method_validation_inputs = {
            'raw': (validation_diff,
                    scores_raw
                    ),
            'z': (validation_diff,
                  scores_z
                  ),
            'page_rank_baseline': (validation_diff,
                                   scores_page_rank
                                   ),
            'random_baseline': (validation_diff,
                                generate_random_score_ranking(kernel)
                                ),
        }

        for method, validation_set in method_validation_inputs.items():
            auroc, auprc = get_metrics(*validation_set)
            auroc_metrics[method].append(auroc)
            auprc_metrics[method].append(auprc)

    return auroc_metrics, auprc_metrics


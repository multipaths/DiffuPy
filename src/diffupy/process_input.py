# -*- coding: utf-8 -*-

"""Main Matrix Class."""

from .matrix import Matrix

def generate_categoric_input_vector_from_labels(rows_labeled, col_label, background_mat, missing_value = -1, rows_unlabeled = None):
    if isinstance(col_label, str):
        col_label = [col_label]

    input_mat = Matrix(rows_labels = list(rows_labeled),
                       cols_labels = col_label,
                       init_value=1)
    if rows_unlabeled:
        input_mat.row_bind(matrix = Matrix(rows_labels = list(rows_unlabeled),
                                           cols_labels = col_label,
                                           init_value = 0)
                          )

    return input_mat.match_missing_rows(background_mat.rows_labels, missing_value).match_rows(background_mat)


def generate_categoric_input_from_labels(rows_labels, cols_labels, background_mat, missing_value = -1, rows_unlabeled = None, ):
    if isinstance(cols_labels, list) and len(cols_labels) > 1:
        input_mat = generate_categoric_input_vector_from_labels(rows_labels[0],
                                                                cols_labels[0],
                                                                background_mat,
                                                                missing_value,
                                                                rows_unlabeled[0])

        for idx, row_label in enumerate(rows_labels[1:]):
            input_vector = generate_categoric_input_vector_from_labels(row_label,
                                                                       cols_labels[idx + 1],
                                                                       background_mat,
                                                                       missing_value,
                                                                       rows_unlabeled[idx + 1],
                                                                       )
            input_mat.col_bind(matrix=input_vector)

        return input_mat
    else:
        return generate_categoric_input_vector_from_labels(rows_labels,
                                                           cols_labels,
                                                           background_mat,
                                                           missing_value,
                                                           rows_unlabeled
                                                           )

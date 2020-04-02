# -*- coding: utf-8 -*-

"""Main matrix class and processing of input data."""
from typing import Dict, List, Optional

import networkx as nx
import pandas as pd

from .constants import *
from .matrix import Matrix

"""Process datasets"""


def process_input(
        path: str,
        method: str,
        binning: bool,
        absolute_value: bool,
        p_value: float,
        threshold: Optional[float],
) -> Dict[str, int]:
    """Read input file and ensure necessary columns exist."""
    if path.endswith(CSV):
        fmt = CSV

    elif path.endswith(TSV):
        fmt = TSV

    else:
        raise IOError(
            f'There is a problem with your file. Please ensure the file you submitted is correctly formatted with a'
            f'.csv or .tsv file extension.'
        )

    df = pd.read_csv(
        path,
        header=0,
        sep=FORMAT_SEPARATOR_MAPPING[CSV] if fmt == CSV else FORMAT_SEPARATOR_MAPPING[TSV]
    )

    # Ensure that column Node is in dataset
    if NODE not in df.columns:
        raise ValueError(
            f'Ensure that your file contains a column {NODE} with node IDs.'
        )

    # If logFC column not in dataFrame, ensure node type column is at least given
    elif LOG_FC not in df.columns:
        if NODE_TYPE not in df.columns:
            raise ValueError(
                f'Ensure that your file contains a column, {NODE_TYPE}, indicating node types.'
            )

    return _codify_input_data(df, method, binning, absolute_value, p_value, threshold)


"""Codify input according to diffusion scoring method"""


def _codify_input_data(
        df: pd.DataFrame,
        method: str,
        binning: bool,
        absolute_value: bool,
        p_value: float,
        threshold: Optional[float],
) -> Dict[str, int]:
    """Prepare input data for diffusion."""
    # Prepare input data for quantitative diffusion scoring methods
    if method == RAW or method == Z:
        return _codify_quantitative_input_data(df, binning, absolute_value, p_value, threshold)

    # Prepare input data for non-quantitative diffusion methods
    elif method == ML or method == GM:
        return _codify_non_quantitative_input_data(df, p_value, threshold)

    else:
        # TODO: ber_s, ber_p, mc
        raise NotImplementedError('This diffusion method has not yet been implemented.')


"""Assign binary labels to input for scoring methods that accept non-quantitative values"""


def _codify_non_quantitative_input_data(
        df: pd.DataFrame,
        p_value: float,
        threshold: Optional[float]
) -> Dict[str, int]:
    """Codify input data to get a set of labelled nodes for scoring methods that accept non-quantitative values."""
    # LogFC provided in dataset and threshold given
    if LOG_FC in df.columns and threshold:

        # Label nodes with 1 if |logFC| passes threshold
        df.loc[(df[LOG_FC]).abs() >= threshold, LABEL] = 1
        # Label nodes with -1 if |logFC| below threshold
        df.loc[(df[LOG_FC]).abs() < threshold, LABEL] = -1

        # If adjusted p-values are provided in dataset, label nodes that are not statistically significant with -1
        if P_VALUE in df.columns:
            df.loc[df[P_VALUE] > p_value, LABEL] = -1

        return df.set_index(NODE)[LABEL].to_dict()

    # If input dataset exclusively contains IDs and no logFC, or if threshold is not given, then assign labels as 1
    df[LABEL] = 1

    return df.set_index(NODE)[LABEL].to_dict()


"""Assign binary labels to input for scoring methods that accept quantitative values"""


def _codify_quantitative_input_data(
        df: pd.DataFrame,
        binning: bool,
        absolute_value: bool,
        p_value: float,
        threshold: Optional[float],
) -> Dict[str, int]:
    """Codify input data to get a set of labelled nodes for scoring methods that accept quantitative values."""
    # LogFC provided in dataset and threshold given
    if LOG_FC in df.columns and threshold:

        # Binarize labels with 1, 0 and/or -1
        if binning is True:

            # Add binning labels where |logFC| values above threshold are 1 and below are 0
            if absolute_value is True:
                return _bin_quantitative_input_by_abs_val(df, threshold, p_value)

            # Add signed labels where |logFC| values above threshold are 1 or -1 (signed) and values below are 0

            return _bin_quantitative_input_by_threshold(df, threshold, p_value)

        # Labels are 0s or logFC values rather than binary values
        else:
            # Codify inputs with |logFC| if they pass threshold; otherwise assign label as 0
            if absolute_value is True:
                return _codify_quantitative_input_by_abs_val(df, threshold, p_value)

            # Codify inputs with logFC if they pass threshold; otherwise assign label as 0
            return _codify_quantitative_input_by_threshold(df, threshold, p_value)

    # If input dataset exclusively contains IDs and no logFC, or if threshold is not given, then assign labels as 1
    df[LABEL] = 1

    # TODO handle NODE_TYPE
    return df.set_index(NODE)[LABEL].to_dict()


def _bin_quantitative_input_by_abs_val(
        df: pd.DataFrame,
        threshold: float,
        p_value: float,
) -> Dict[str, int]:
    """Process quantitative inputs and bin labels by absolute value."""
    # Add label 1 if |logFC| is above threshold
    df.loc[(df[LOG_FC]).abs() >= threshold, LABEL] = 1
    # Add label 0 if |logFC| below threshold
    df.loc[(df[LOG_FC]).abs() < threshold, LABEL] = 0

    # logFC and adjusted p-values are provided in dataset
    if P_VALUE in df.columns:
        return _remove_non_significant_entities(df, p_value)

    return df.set_index(NODE)[LABEL].to_dict()


def _bin_quantitative_input_by_threshold(
        df: pd.DataFrame,
        threshold: float,
        p_value: float,
) -> Dict[str, int]:
    """Process quantitative inputs and bin labels by threshold."""
    # Add label 1 if logFC is above threshold
    df.loc[df[LOG_FC] >= threshold, LABEL] = 1
    # Add label 0 if |logFC| below threshold
    df.loc[(df[LOG_FC]).abs() < threshold, LABEL] = 0
    # Replace remaining labels with -1 (i.e. |logFC| above threshold but sign is negative)
    df = df.fillna(-1)

    if p_value:
        # LogFC values and adjusted p-values are provided in dataset
        if P_VALUE in df.columns:
            # Disregard entities if logFC adjusted p-value is not significant
            return _remove_non_significant_entities(df, p_value)

    return df.set_index(NODE)[LABEL].to_dict()


"""Assign logFC as labels for input for scoring methods that accept quantitative values"""


def _codify_quantitative_input_by_abs_val(
        df: pd.DataFrame,
        threshold: float,
        p_value: float,
) -> Dict[str, int]:
    """Codify nodes with |logFC| if they pass threshold, otherwise label is 0."""
    # Codify nodes with |logFC| if they pass threshold
    df.loc[(df[LOG_FC]).abs() >= threshold, LABEL] = (df[LOG_FC]).abs()
    # Codify nodes with label 0 if it falls below threshold
    df.loc[(df[LOG_FC]).abs() < threshold, LABEL] = 0

    # LogFC and adjusted p-values are provided in dataset
    if P_VALUE in df.columns:
        # Disregard entities if logFC adjusted p-value is not significant
        return _remove_non_significant_entities(df, p_value)

    return df.set_index(NODE)[LABEL].to_dict()


def _codify_quantitative_input_by_threshold(
        df: pd.DataFrame,
        threshold: float,
        p_value: float,
) -> Dict[str, int]:
    """Codify inputs with logFC if they pass threshold value."""
    df.loc[df[LOG_FC] >= threshold, LABEL] = df[LOG_FC]
    df.loc[(df[LOG_FC]).abs() < threshold, LABEL] = 0
    df.loc[((df[LOG_FC]).abs() >= threshold) & ((df[LOG_FC]) < 0), LABEL] = df[LOG_FC]

    # LogFC values and adjusted p-values are provided in dataset
    if P_VALUE in df.columns:
        # Disregard entities if logFC adjusted p-value is not significant
        return _remove_non_significant_entities(df, p_value)

    return df.set_index(NODE)[LABEL].to_dict()


def _remove_non_significant_entities(df: pd.DataFrame, p_value: float) -> pd.DataFrame:
    # Label entity 0 if adjusted p-value for logFC is not significant
    df.loc[df[P_VALUE] > p_value, LABEL] = 0

    return df.set_index(NODE)[LABEL].to_dict()


"""Map nodes from input to network"""


def map_nodes(input_node_dict: Dict[str, int], network: nx.Graph) -> List:
    """Map nodes from input dataset to nodes in network to get a set of labelled and unlabelled nodes."""
    # List of nodes in network
    network_nodes = list(network.nodes)

    return [input_node_dict[node] if node in input_node_dict else None for node in network_nodes]


"""Generate input vector from dataset labels"""


def generate_categoric_input_vector_from_labels(
        rows_labeled,
        col_label,
        background_mat,
        missing_value=-1,
        rows_unlabeled=None,
):
    """Generate categoric input vector from labels."""
    if isinstance(col_label, str):
        col_label = [col_label]

    input_mat = Matrix(
        rows_labels=list(rows_labeled),
        cols_labels=col_label,
        init_value=1)
    if rows_unlabeled:
        input_mat.row_bind(
            matrix=Matrix(
                rows_labels=list(rows_unlabeled),
                cols_labels=col_label,
                init_value=0)
        )

    return input_mat.match_missing_rows(background_mat.rows_labels, missing_value).match_rows(background_mat)


def generate_categoric_input_from_labels(
        rows_labels,
        cols_labels,
        background_mat,
        missing_value=-1,
        rows_unlabeled=None,
):
    """Generate input vector from labels."""
    if isinstance(cols_labels, list) and len(cols_labels) > 1:
        input_mat = generate_categoric_input_vector_from_labels(
            rows_labels[0],
            cols_labels[0],
            background_mat,
            missing_value,
            rows_unlabeled[0]
        )

        for idx, row_label in enumerate(rows_labels[1:]):
            input_vector = generate_categoric_input_vector_from_labels(
                row_label,
                cols_labels[idx + 1],
                background_mat,
                missing_value,
                rows_unlabeled[idx + 1],
            )
            input_mat.col_bind(matrix=input_vector)

        return input_mat
    else:
        return generate_categoric_input_vector_from_labels(
            rows_labels,
            cols_labels,
            background_mat,
            missing_value,
            rows_unlabeled
        )

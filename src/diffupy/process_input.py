# -*- coding: utf-8 -*-

"""Main matrix class and processing of input data."""

import logging
from collections import defaultdict
from typing import Dict, Optional, Union, List, Set, Tuple

import numpy as np
import pandas as pd

from .constants import *
from .matrix import Matrix
from .utils import from_pickle, from_json, from_dataframe_file, from_nparray_to_df, get_random_value_from_dict, \
    get_random_key_from_dict, parse_xls_to_df, print_dict_dimensions

log = logging.getLogger(__name__)

"""Process input data"""


def process_map_and_format_input_data_for_diff(
        data_input: Union[str, pd.DataFrame, list, dict, np.ndarray, Matrix],
        kernel: Matrix,
        method: str = 'raw',
        binning: Optional[bool] = False,
        absolute_value: Optional[bool] = False,
        p_value: Optional[float] = None,
        threshold: Optional[float] = None,
        background_labels: Optional[Union[list, Dict[str, list]]] = None,
        show_statistics: bool = True,
        **further_parse_args
) -> Matrix:
    """Process miscellaneous data input, perform the mapping to the diffusion background network (as a kernel) and format it for the diffusion computation function.

    :param data_input: A miscellaneous data input to be processed/formatted for the diffuPy diffusion computation.
    :param kernel: A pre-computed kernel to perform the label mapping and the matching for the input formatting.
    :param method: Elected method ["raw", "ml", "gm", "ber_s", "ber_p", "mc", "z"].
    :param binning: If logFC provided in dataset, convert logFC to binary.
    :param absolute_value: Codify node labels by applying threshold to | logFC | in input.
    :param p_value: Statistical significance.
    :param threshold: Codify node labels by applying a threshold to logFC in input.
    :param background_labels: Labels set to map the input labels, which can provide label classification by type dict.
    :param further_parse_args: Arguments to refine the data input parsing, among which:
                                for string list parsing: separ_str
                                for excel/csv parsing: min_row, cols_mapping, relevant_cols, irrelevant_cols
                                for excel: relevant_sheets, irrelevant_sheets
                                for mapping: check_substrings (as a bool if input list or list of labels types if input dict)
    """
    # If specific label background not provided, get a list from kernel labels.
    if not background_labels:
        background_labels = list(kernel.rows_labels)

    # Pipeline the input, first preprocessing it, then mapping it to the background labels
    # and finally formatting it with the kernel reference.
    return format_input_for_diffusion(map_labels_input(input_labels=process_input_data(data_input,
                                                                                       method,
                                                                                       binning,
                                                                                       absolute_value,
                                                                                       p_value,
                                                                                       threshold,
                                                                                       **further_parse_args
                                                                                       ),
                                                       background_labels=background_labels,
                                                       check_substrings=further_parse_args.get('check_substrings'),
                                                       show_descriptive_stat=show_statistics
                                                       ),
                                      kernel
                                      )


def process_input_data(
        data_input: Union[str, list, dict, np.ndarray, pd.DataFrame],
        method: str = RAW,
        binning: bool = False,
        absolute_value: bool = False,
        p_value: float = 0.05,
        threshold: Optional[float] = 0.5,
        **further_parse_args
) -> Union[list, Dict[str, int], Dict[str, Dict[str, int]], Dict[str, list]]:
    """Pipeline the provided miscellaneous data input for further processing, in the following standardized data structures: label list, type_dict label lists, label-scores dict or type_dict label-scores dicts.

    :param data_input: A miscellaneous data input to be processed.
    :param method: Elected method ["raw", "ml", "gm", "ber_s", "ber_p", "mc", "z"]
    :param binning: If logFC provided in dataset, convert logFC to binary.
    :param absolute_value: Codify node labels by applying threshold to | logFC | in input.
    :param p_value: Statistical significance.
    :param threshold: Codify node labels by applying a threshold to logFC in input.
    :param further_parse_args: Arguments to refine the data input parsing, among which:
                                for string list parsing: separ_str
                                for excel/csv parsing: min_row, cols_mapping, relevant_cols, irrelevant_cols
                                for excel: relevant_sheets, irrelevant_sheets
    """
    log.info("Processing the data input.")

    # Preprocess the raw input according its data structure types.
    preprocessed_data = _process_data_input_format(data_input, **further_parse_args)

    # If the preprocessed input is a list or a label type dict (Dict[str, list]) return it for latter categorical input generation.
    if _label_list_data_struct_check(preprocessed_data) or _type_dict_label_list_data_struct_check(preprocessed_data):
        return preprocessed_data

    # If the preprocessed input is a label type label-scores dict (Dict[str, pd.DataFrame]) pipeline it for scores codifying.
    if isinstance(preprocessed_data, dict):
        return {label_type: _codify_input_data(preprocessed_data_i,
                                               method,
                                               binning,
                                               absolute_value,
                                               p_value,
                                               threshold,
                                               further_parse_args.get('cols_titles_mapping')
                                               )
                for label_type, preprocessed_data_i in preprocessed_data.items()
                }

    # If the preprocessed input is a scores-label dataframe (pd.DataFrame) pipeline it for scores codifying.
    return _codify_input_data(preprocessed_data,
                              method,
                              binning,
                              absolute_value,
                              p_value,
                              threshold,
                              further_parse_args.get('cols_titles_mapping')
                              )


"""Process input formats"""


def _process_data_input_format(
        raw_data_input: Union[str, list, dict, np.ndarray, pd.DataFrame],
        separ_str: str = ', ',
        **further_parse_args
) -> Union[pd.DataFrame, list, Dict[str, Union[pd.DataFrame, list]]]:
    """Format the input as a label-score dataframe, a list or a labels or a type dict for latter input processing."""
    if isinstance(raw_data_input, str):
        # If the data input type is a string, mostly will be a path to the dataset file.
        if os.path.isfile(raw_data_input):
            return _process_data_input_format(_load_data_input_from_file(raw_data_input, **further_parse_args))
        elif '/' in raw_data_input and separ_str not in ['/', ' /', '/ ']:
            raise IOError(
                f'{EMOJI} The file could not have been located in the provided data input path,.'
            )
        # If the data input is not identified as a path, it will be treated as a label list with an indicated separator.
        else:
            return _process_data_input_format(raw_data_input.split(separ_str))

    elif isinstance(raw_data_input, list) or isinstance(raw_data_input, set):
        return list(set(raw_data_input))

    if isinstance(raw_data_input, pd.DataFrame):
        return raw_data_input

    elif isinstance(raw_data_input, dict):
        # If the data input type dict is a label-scores dict, codify it as a Panda's dataframe for latter processing.
        if _label_scores_dict_data_struct_check(raw_data_input):
            df = pd.DataFrame.from_dict({NODE: list(raw_data_input.keys()), LOG_FC: list(raw_data_input.values())})
            return df
        # Else it will be treated as a label_type dict, calling recursively the process input format for each type subset (key).
        else:
            # It is assumed that the all the dict values match the same data type.
            return {label_type: _process_data_input_format(data_i) for label_type, data_i in raw_data_input.items()}

    elif isinstance(raw_data_input, np.ndarray):
        return from_nparray_to_df(raw_data_input)

    elif isinstance(raw_data_input, Matrix):
        return raw_data_input.to_df()

    else:
        raise TypeError(
            f'{EMOJI} The imported kernel type is not valid. Please ensure is provided as a diffupy '
            f'Matrix, a Dict, NumpyArray or Pandas DataFrame. '
        )


def _load_data_input_from_file(path: str, **further_parse_args) -> Union[pd.DataFrame, list]:
    """Load and process the input data according the input file format."""
    if path.endswith(CSV):
        return from_dataframe_file(path, CSV)

    elif path.endswith(XLS_FORMATS):
        return parse_xls_to_df(path,
                               further_parse_args.get('min_row'),
                               further_parse_args.get('relevant_sheets'),
                               further_parse_args.get('irrelevant_sheets'),
                               further_parse_args.get('relevant_cols'),
                               further_parse_args.get('irrelevant_cols')
                               )

    elif path.endswith(TSV):
        return from_dataframe_file(path, TSV)

    elif path.endswith(PICKLE):
        return from_pickle(path)

    elif path.endswith(JSON):
        return from_json(path)

    else:
        raise IOError(
            'There is a problem with your file. Please ensure the file you submitted is correctly formatted with a'
            '.csv or .tsv file extension.'
        )


"""Pipeline input scores"""


def _codify_input_data(
        df: pd.DataFrame,
        method: str,
        binning: bool,
        absolute_value: bool,
        p_value: float,
        threshold: Optional[float],
        cols_titles_mapping: Optional[Dict[str, str]] = None
) -> Union[Dict[str, Dict[str, int]], Dict[str, int]]:
    """Process the input scores dataframe for the codifying process."""
    # Rename dataframe column titles according (if) provided label_mapping.
    if cols_titles_mapping is not None:
        for label_to_rename, new_name in cols_titles_mapping.items():
            if label_to_rename in df.columns:
                df = df.rename(columns={label_to_rename: new_name})

    # Ensure that node labeling is in the provided dataset.
    if not any(n in df.columns for n in NODE_LABELING):
        raise ValueError(
            f'Ensure that your file contains a column {NODE_LABELING} with node IDs.'
        )

    # Standardize the title of the node column labeling column to 'Label', for later processing.
    if LABEL not in df.columns:
        for column_label in list(df.columns):
            if column_label in NODE_LABELING:
                df = df.rename(columns={column_label: LABEL})
                break

    # If node type provided in a column, classify in a dictionary the input codification by its node type.
    if NODE_TYPE in df.columns:

        node_types = list(set(df[NODE_TYPE]))  # Get the node types list set.
        codified_by_type_dict = {}

        for node_type in node_types:
            # Filter the nodes by the iterable type.
            df_by_type = df.loc[df[NODE_TYPE] == node_type]

            # Codify the nodes for the iterable type.
            codified_by_type_dict[node_type] = _codify_method_check(df_by_type,
                                                                    method,
                                                                    binning,
                                                                    absolute_value,
                                                                    p_value,
                                                                    threshold
                                                                    )
        return codified_by_type_dict

    else:
        # Codify all the nodes of the dataframe.
        return _codify_method_check(df,
                                    method,
                                    binning,
                                    absolute_value,
                                    p_value,
                                    threshold
                                    )


def _codify_method_check(
        df: pd.DataFrame,
        method: str,
        binning: bool,
        absolute_value: bool,
        p_value: float,
        threshold: Optional[float],
) -> Dict[str, int]:
    """Classify the input data codification according the diffusion method."""
    # Prepare input data for quantitative diffusion scoring methods
    if method == RAW or method == Z:
        return _codify_quantitative_input_data(df, binning, absolute_value, p_value, threshold)

    # Prepare input data for non-quantitative diffusion methods
    elif method == ML or method == GM:
        return _codify_non_quantitative_input_data(df, p_value, threshold)

    else:
        # TODO: ber_s, ber_p, mc
        raise NotImplementedError('This diffusion method has not been yet implemented.')


"""Assign binary scores to input for scoring methods that ONLY accept non-quantitative values"""


def _codify_non_quantitative_input_data(
        df: pd.DataFrame,
        p_value: float,
        threshold: Optional[float]
) -> Dict[str, int]:
    """Codify input data to get a set of scored nodes for scoring methods that accept non-quantitative values."""
    # LogFC provided in dataset and threshold given
    if LOG_FC in df.columns and threshold:

        # Label nodes with 1 if | logFC | passes threshold
        df.loc[(df[LOG_FC]).abs() >= threshold, SCORE] = 1
        # Label nodes with -1 if | logFC | below threshold
        df.loc[(df[LOG_FC]).abs() < threshold, SCORE] = -1

        # If adjusted p-values are provided in dataset, score nodes that are not statistically significant with -1
        if P_VALUE in df.columns:
            df.loc[df[P_VALUE] > p_value, SCORE] = -1

        return df.set_index(LABEL)[SCORE].to_dict()

    # If input dataset exclusively contains IDs and no logFC, or if threshold is not given, then assign scores as 1
    df[SCORE] = 1

    return df.set_index(LABEL)[SCORE].to_dict()


"""Assign binary scores to input for scoring methods that accept quantitative values"""


def _codify_quantitative_input_data(
        df: pd.DataFrame,
        binning: bool,
        absolute_value: bool,
        p_value: float,
        threshold: Optional[float],
) -> Dict[str, int]:
    """Codify input data to get a set of scored nodes for scoring methods that accept quantitative values."""
    # LogFC provided in dataset and threshold given
    if LOG_FC in df.columns and threshold:

        # Binarize scores with 1, 0 and/or -1
        if binning is True:

            # Add binning scores where | logFC | values above threshold are 1 and below are 0
            if absolute_value is True:
                return _bin_quantitative_input_by_abs_val(df, threshold, p_value)

            # Add signed scores where | logFC | values above threshold are 1 or -1 (signed) and values below are 0

            return _bin_quantitative_input_by_threshold(df, threshold, p_value)

        # Labels are 0s or logFC values rather than binary values
        else:
            # Codify inputs with | logFC | if they pass threshold; otherwise assign score as 0
            if absolute_value is True:
                return _codify_quantitative_input_by_abs_val(df, threshold, p_value)

            # Codify inputs with logFC if they pass threshold; otherwise assign score as 0
            return _codify_quantitative_input_by_threshold(df, threshold, p_value)

    # If input dataset exclusively contains IDs and no logFC, or if threshold is not given, then assign scores as 1
    df[SCORE] = 1

    return df.set_index(LABEL)[SCORE].to_dict()


def _bin_quantitative_input_by_abs_val(
        df: pd.DataFrame,
        threshold: float,
        p_value: float,
) -> Dict[str, int]:
    """Process quantitative inputs and bin scores by absolute value."""
    # Add score 1 if | logFC | is above threshold
    df.loc[(df[LOG_FC]).abs() >= threshold, SCORE] = 1
    # Add score 0 if | logFC | below threshold
    df.loc[(df[LOG_FC]).abs() < threshold, SCORE] = 0

    # logFC and adjusted p-values are provided in dataset
    if P_VALUE in df.columns:
        return _remove_non_significant_entities(df, p_value)

    return df.set_index(LABEL)[SCORE].to_dict()


def _bin_quantitative_input_by_threshold(
        df: pd.DataFrame,
        threshold: float,
        p_value: float,
) -> Dict[str, int]:
    """Process quantitative inputs and bin scores by threshold."""
    # Add score 1 if logFC is above threshold
    df.loc[df[LOG_FC] >= threshold, SCORE] = 1
    # Add score 0 if | logFC | below threshold
    df.loc[(df[LOG_FC]).abs() < threshold, SCORE] = 0
    # Replace remaining score with -1 (i.e. | logFC | above threshold but sign is negative)
    df = df.fillna(-1)

    if p_value:
        # LogFC values and adjusted p-values are provided in dataset
        if P_VALUE in df.columns:
            # Disregard entities if logFC adjusted p-value is not significant
            return _remove_non_significant_entities(df, p_value)

    return df.set_index(LABEL)[SCORE].to_dict()


"""Assign logFC as score for input for scoring methods that accept quantitative values"""


def _codify_quantitative_input_by_abs_val(
        df: pd.DataFrame,
        threshold: float,
        p_value: float,
) -> Dict[str, int]:
    """Codify nodes with | logFC | if they pass threshold, otherwise score is 0."""
    # Codify nodes with | logFC | if they pass threshold
    df.loc[(df[LOG_FC]).abs() >= threshold, SCORE] = (df[LOG_FC]).abs()
    # Codify nodes with score 0 if it falls below threshold
    df.loc[(df[LOG_FC]).abs() < threshold, SCORE] = 0

    # LogFC and adjusted p-values are provided in dataset
    if P_VALUE in df.columns:
        # Disregard entities if logFC adjusted p-value is not significant
        return _remove_non_significant_entities(df, p_value)

    return df.set_index(LABEL)[SCORE].to_dict()


def _codify_quantitative_input_by_threshold(
        df: pd.DataFrame,
        threshold: float,
        p_value: float,
) -> Dict[str, int]:
    """Codify inputs with logFC if they pass threshold value."""
    df.loc[df[LOG_FC] >= threshold, SCORE] = df[LOG_FC]
    df.loc[(df[LOG_FC]).abs() < threshold, SCORE] = 0
    df.loc[((df[LOG_FC]).abs() >= threshold) & ((df[LOG_FC]) < 0), SCORE] = df[LOG_FC]

    # LogFC values and adjusted p-values are provided in dataset
    if P_VALUE in df.columns:
        # Disregard entities if logFC adjusted p-value is not significant
        return _remove_non_significant_entities(df, p_value)

    return df.set_index(LABEL)[SCORE].to_dict()


def _remove_non_significant_entities(df: pd.DataFrame, p_value: float) -> Dict[str, int]:
    # Label entity 0 if adjusted p-value for logFC is not significant
    df.loc[df[P_VALUE] > p_value, SCORE] = 0

    return df.set_index(LABEL)[SCORE].to_dict()


"""Data structures format checkers"""


def _label_scores_dict_data_struct_check(v: Union[dict, list]) -> bool:
    """Check data structure type Dict[str, int]."""
    return isinstance(v, dict) and isinstance(get_random_value_from_dict(v), (int, float))


def _type_dict_label_scores_dict_data_struct_check(v: Union[dict, list]) -> bool:
    """Check data structure type Dict[str, Dict[str, int]]."""
    return isinstance(v, dict) and isinstance(get_random_value_from_dict(v), dict) and isinstance(
        get_random_value_from_dict(get_random_value_from_dict(v)), (int, float))


def _two_dimensional_type_dict_label_scores_dict_data_struct_check(v: Union[dict, list]) -> bool:
    """Check data structure type Dict[str, Dict[str, Dict[str, int]]]."""
    return isinstance(v, dict) and isinstance(get_random_value_from_dict(v), dict) and isinstance(
        get_random_value_from_dict(get_random_value_from_dict(v)), dict) and isinstance(
        get_random_value_from_dict(get_random_value_from_dict(get_random_value_from_dict(v))), (int, float))


def _label_list_data_struct_check(v: Union[dict, list]) -> bool:
    """Check data structure type list."""
    return isinstance(v, list) or isinstance(v, set)


def _type_dict_label_list_data_struct_check(v: Union[dict, list]) -> bool:
    """Check data structure type Dict[str, list]."""
    return isinstance(v, dict) and _label_list_data_struct_check(get_random_value_from_dict(v))


def _two_dimensional_type_dict_label_list_data_struct_check(v: Union[dict, list]) -> bool:
    """Check data structure type Dict[str, Dict[str, list]]."""
    return isinstance(v, dict) and isinstance(get_random_value_from_dict(v), dict) and _label_list_data_struct_check(
        get_random_value_from_dict(get_random_value_from_dict(v)))


"""Mappers from input to network background"""


def map_labels_input(
        input_labels: Union[list, Dict[str, int], Dict[str, list], Dict[str, Dict[str, int]]],
        background_labels: Union[list, Dict[str, list], Dict[str, Dict[str, list]]],
        check_substrings: Union[List, bool] = None,
        show_descriptive_stat: bool = False
) -> Union[Dict[str, int], list]:
    """Get the mappings from preprocessed input_labels."""
    log.info("Mapping the input labels to the background labels reference.")

    if _label_list_data_struct_check(background_labels):
        mapped_labels = _map_labels_to_background(input_labels,
                                                  background_labels,
                                                  check_substring=check_substrings)

    # If type dict _map_labels_to_background for each classified input_labels.
    elif _type_dict_label_list_data_struct_check(background_labels) or _type_dict_label_scores_dict_data_struct_check(
            background_labels):
        mapped_labels = {node_type: _map_labels_to_background(input_labels,
                                                              node_set,
                                                              background_labels_type=node_type,
                                                              check_substring=check_substrings)
                         for node_type, node_set
                         in background_labels.items()
                         if _map_labels_to_background(input_labels,
                                                      node_set,
                                                      background_labels_type=node_type,
                                                      check_substring=check_substrings) not in [[], {}]
                         }

    # If two-dimensional type dict call recursively map_labels_input.
    elif _two_dimensional_type_dict_label_list_data_struct_check(
            background_labels) or _two_dimensional_type_dict_label_scores_dict_data_struct_check(background_labels):
        mapped_labels = {node_type: map_labels_input(input_labels,
                                                     node_set,
                                                     check_substrings,
                                                     show_descriptive_stat=False
                                                     )
                         for node_type, node_set
                         in background_labels.items()
                         if map_labels_input(input_labels,
                                             node_set,
                                             check_substrings,
                                             show_descriptive_stat=False
                                             ) not in [[], {}]
                         }

    else:
        raise IOError(
            f'{EMOJI} The background mapping labels should be provided as a label list or as a type dict of label list.'
        )

    if show_descriptive_stat:
        print_dict_dimensions(mapping_statistics(input_labels, mapped_labels), title='Mapping descriptive statistics')

    return mapped_labels


def mapping_statistics(
        input_labels: Union[list, Dict[str, Dict[str, list]], Dict[str, list]],
        mapped_labels: Union[list, Dict[str, Dict[str, list]], Dict[str, list]],
        subtotals: Dict[str, int] = None
) -> Dict:
    """Calculate mapping descriptive statistics."""
    statistics_dict = {}

    total_mapping = set()
    total_input = set()

    if _label_list_data_struct_check(mapped_labels) or _label_scores_dict_data_struct_check(mapped_labels):
        total_mapping = mapped_labels
        total_input = input_labels
        if len(total_input) != 0:
            statistics_dict['total'] = (len(total_mapping), len(total_mapping) / len(total_input))

    elif _type_dict_label_list_data_struct_check(mapped_labels) or _type_dict_label_scores_dict_data_struct_check(
            mapped_labels):
        for mapping_type, mapping in mapped_labels.items():

            if (_type_dict_label_list_data_struct_check(input_labels) or _type_dict_label_scores_dict_data_struct_check(
                    input_labels)) and mapping_type in input_labels.keys():
                if len(input_labels[mapping_type]) != 0:
                    statistics_dict[mapping_type] = (len(mapping), len(mapping) / len(input_labels[mapping_type]))
                else:
                    statistics_dict[mapping_type] = (0, 0)

                total_mapping.update(mapping)
                total_input.update(input_labels[mapping_type])
            else:
                if subtotals is None:
                    subtotal_input = len(input_labels)
                else:
                    subtotal_input = subtotals[mapping_type]

                if subtotal_input != 0:
                    statistics_dict[mapping_type] = (len(mapping), len(mapping) / subtotal_input)
                else:
                    statistics_dict[mapping_type] = (0, 0)

                total_input.update(input_labels)

            total_mapping.update(mapping)

        if subtotals:
            statistics_dict['total_mapping'] = total_mapping
            statistics_dict['total_input'] = total_input

        if len(total_input) != 0:
            statistics_dict['total'] = (len(total_mapping), len(total_mapping) / len(total_input))

    elif _two_dimensional_type_dict_label_scores_dict_data_struct_check(
            mapped_labels) or _two_dimensional_type_dict_label_list_data_struct_check(mapped_labels):

        subtotals_dict = defaultdict(set)

        for _, mapping_subdict in mapped_labels.items():
            for mapping_subtype, mapping_subdict in mapping_subdict.items():
                subtotals_dict[mapping_subtype].update(mapping_subdict)

        subtotals_dict = {
            mapping_subtype: len(mapping_subdict)
            for mapping_subtype, mapping_subdict in
            subtotals_dict.items()
        }

        for mapping_type, mapping_subdict in mapped_labels.items():
            percentage_dict_i = mapping_statistics(input_labels, mapping_subdict, subtotals=subtotals_dict)

            statistics_dict[mapping_type] = percentage_dict_i

            total_mapping.update(percentage_dict_i.pop('total_mapping'))
            total_input.update(percentage_dict_i.pop('total_input'))

        if len(total_input) != 0:
            subtotals_dict = {mapping_type: (mapping, mapping / len(total_input)) for mapping_type, mapping in
                              subtotals_dict.items()}
            subtotals_dict['total'] = (len(total_mapping), len(total_mapping) / len(total_input))

            statistics_dict['total'] = subtotals_dict

    else:
        raise TypeError(
            f'{EMOJI} The input labels data structure can not be processed for label mapping'
        )

    if len(total_input) == 0:
        statistics_dict['total'] = (0, 0)

    return statistics_dict


def _map_labels(
        input_labels: Union[list, Dict[str, Dict[str, int]], Dict[str, int], Dict[str, list]],
        background_labels: list,
        check_substrings: bool = False
) -> Union[list, Dict[str, Dict[str, int]], Dict[str, int], Dict[str, list]]:
    """Map nodes from input dataset to nodes in network to get a set of labelled and unlabelled nodes."""
    if _label_list_data_struct_check(input_labels):
        return _map_label_list(input_labels, background_labels, check_substrings)

    elif _label_scores_dict_data_struct_check(input_labels):
        return _map_label_dict(input_labels, background_labels, check_substrings)

    elif _type_dict_label_list_data_struct_check(input_labels):
        map_list = []
        for type, label_list in input_labels.items():
            map_list += _map_labels(label_list, background_labels, check_substrings)
        return map_list

    elif _type_dict_label_scores_dict_data_struct_check(input_labels):
        map_dict = {}
        for type, scores_dict in input_labels.items():
            map_dict.update(_map_labels(scores_dict, background_labels, check_substrings))
        return map_dict

    else:
        raise TypeError(
            f'{EMOJI} The input labels data structure can not be processed for label mapping'
        )


def _map_labels_to_background(
        input_labels: Union[list, Dict[str, Dict[str, int]], Dict[str, int], Dict[str, list]],
        background_labels: list,
        background_labels_type: str = None,
        check_substring: Union[List, bool] = None
) -> Union[Dict[str, Dict[str, int]], Dict[str, int]]:
    """Map labels from preprocessed input to background_labels to get a set of matched labels."""
    if _type_dict_label_scores_dict_data_struct_check(input_labels) or \
            _type_dict_label_list_data_struct_check(input_labels):

        if background_labels_type and background_labels_type in input_labels.keys():
            return _map_labels(input_labels[background_labels_type], background_labels,
                               check_substring is not None and background_labels_type in check_substring)
        return {
            type: _map_labels(label_list, background_labels,
                              check_substring is not None and type in check_substring)
            for type, label_list in input_labels.items()
            if _map_labels(label_list, background_labels,
                           check_substring is not None and type in check_substring) not in [[], {}]
        }

    return _map_labels(input_labels, background_labels, check_substring)


def _check_label_to_background_labels(
        label: str,
        label_list: List[Union[str, Tuple[str]]],
        substring: bool = False
) -> Union[str, None]:
    """Check if label string in a label list, also check further if substring checking."""
    if label in label_list:
        return label

    # If the first fast mapping check do not match, perform further mapping iteration
    for entity in label_list:

        if isinstance(entity, set) or isinstance(entity, tuple) or isinstance(entity, list):
            for subentity in entity:
                if not substring:
                    if str(subentity) == label:
                        return subentity
                elif str(subentity) in label or label in str(subentity):
                    return subentity

        elif substring and (str(entity) in label or label in str(entity)):
            return entity

    return None


def _map_label_list(
        input_labels: Union[str, Set[str], List[str]],
        background_labels: List[str],
        check_substrings: bool = False
) -> List[str]:
    """Map labels from preprocessed input to background_labels LIST to get a set of matched labels."""
    mapped_list = []
    for label in input_labels:
        if isinstance(label, str):
            label_bck = _check_label_to_background_labels(label, background_labels, check_substrings)
            if label_bck is not None:
                mapped_list.append(label_bck)
        elif isinstance(label, set) or isinstance(label, tuple) or isinstance(label, list):
            for sublabel in set(label):
                label_bck = _check_label_to_background_labels(sublabel, background_labels, check_substrings)
                if label_bck is not None:
                    mapped_list.append(label_bck)
        else:
            raise TypeError(
                f'{EMOJI} The input label "{label}" "{type(label)}" data type can not be processed for label mapping'
            )
    return mapped_list


def _map_label_dict(
        input_labels: Dict[Union[str, set], Union[int, float]],
        background_labels: list,
        check_substrings: bool = False
) -> Dict[str, Union[int, float]]:
    """Map labels from preprocessed input to background_labels DICT to get a set of matched labels."""
    mapped_dict = {}

    for label, v in input_labels.items():
        if isinstance(label, int) or isinstance(label, float):
            label = str(label)

        if isinstance(label, str):
            label_bck = _check_label_to_background_labels(label, background_labels, check_substrings)
            if label_bck is not None:
                mapped_dict[label_bck] = v

        elif isinstance(label, set) or isinstance(label, tuple) or isinstance(label, list):
            for sublabel in set(label):
                label_bck = _check_label_to_background_labels(sublabel, background_labels, check_substrings)
                if label_bck is not None:
                    mapped_dict[label_bck] = v
        else:
            raise TypeError(
                f'{EMOJI} The input label "{label}" "{type(label)}" data type can not be processed for label mapping'
            )

    return mapped_dict


"""Generate/format data input as a vector/matrix for the diffusion computation matching the kernel rows"""


def format_input_for_diffusion(
        processed_input: Union[list, Dict[str, int], Dict[str, Dict[str, int]], Dict[str, list]],
        kernel: Matrix,
        missing_value: int = -1,
        title=''
) -> Matrix:
    """Format/generate input vector/matrix according the data structure of the processed_data_input."""
    log.info("Formatting the processed to the reference kernel Matrix.")

    if _label_list_data_struct_check(processed_input):
        return format_categorical_input_vector_from_label_list(rows_labeled=processed_input,
                                                               col_label='scores',
                                                               kernel=kernel,
                                                               missing_value=missing_value,
                                                               title=title
                                                               )

    elif _type_dict_label_list_data_struct_check(processed_input):
        return format_categorical_input_matrix_from_label_list(rows_labels=list(processed_input.values()),
                                                               cols_labels=list(processed_input.keys()),
                                                               kernel=kernel,
                                                               missing_value=missing_value,
                                                               title=title
                                                               )

    elif _label_scores_dict_data_struct_check(processed_input):
        return format_input_vector_from_label_score_dict(labels_scores_dict=processed_input,
                                                         kernel=kernel,
                                                         missing_value=missing_value,
                                                         title=title
                                                         )

    elif _type_dict_label_scores_dict_data_struct_check(processed_input):
        return format_input_matrix_from_type_label_score_dict(type_dict_labels_scores_dict=processed_input,
                                                              kernel=kernel,
                                                              missing_value=missing_value,
                                                              title=title
                                                              )

    else:
        raise TypeError(
            f'{EMOJI} The label/scores mapping data structure can not be processed for the input formatting.'
        )


"""Generate categorical (non-quantitative) input vector matrix from raw input dataset labels"""


def format_categorical_input_vector_from_label_list(
        rows_labeled: Union[set, list],
        col_label: Union[str, set, list],
        kernel: Matrix,
        missing_value: int = -1,
        rows_unlabeled=None,
        i: int = None,
        title=''
) -> Matrix:
    """Generate categoric input vector from labels."""
    if isinstance(col_label, str):
        col_label = [col_label]

    input_mat = Matrix(
        rows_labels=list(set(rows_labeled)),
        cols_labels=col_label,
        init_value=1,  # By default the categorical labeled input value is 1
        name=title
    )

    if rows_unlabeled:
        if i:
            rows_unlabeled = rows_unlabeled[i]

        input_mat.row_bind(
            matrix=Matrix(
                rows_labels=list(rows_unlabeled),
                cols_labels=col_label,
                init_value=0,  # By default the non labeled input value is 0
            )
        )

    return input_mat.match_delete_rows(kernel.rows_labels).match_missing_rows(kernel.rows_labels,
                                                                              missing_value).match_rows(kernel)


def format_categorical_input_matrix_from_label_list(
        rows_labels: Union[set, list],
        cols_labels: Union[set, list],
        kernel: Matrix,
        missing_value: int = -1,
        rows_unlabeled=None,
        title=''
) -> Matrix:
    """Generate input vector from labels."""
    if not isinstance(cols_labels, list):
        raise NotImplementedError('The column labels should be provided as a list.')

    if len(cols_labels) > 1:

        input_mat = format_categorical_input_vector_from_label_list(
            rows_labels[0],
            cols_labels[0],
            kernel,
            missing_value,
            rows_unlabeled,
            i=0,
            title=title
        )

        for idx, row_label in enumerate(rows_labels[1:]):
            input_vector = format_categorical_input_vector_from_label_list(
                row_label,
                cols_labels[idx + 1],
                kernel,
                missing_value,
                rows_unlabeled,
                idx + 1
            )
            input_mat.col_bind(matrix=input_vector)

        return input_mat

    return format_categorical_input_vector_from_label_list(
        rows_labels,
        cols_labels,
        kernel,
        missing_value,
        rows_unlabeled,
        title=title
    )


"""Generate quantitative or binarized/categorical input vector matrix from preprocesed input dataset scores"""


def format_input_vector_from_label_score_dict(
        labels_scores_dict: Dict[str, int],
        kernel: Matrix,
        col_label: str = 'scores',
        missing_value: int = -1,
        rows_unlabeled: dict = None,  # TODO: To discuss
        type_k: bool = False,
        title=''
) -> Matrix:
    """Generate scores input vector from labels scores dict."""
    input_mat = Matrix(
        mat=np.transpose(np.array([list(labels_scores_dict.values())])),
        rows_labels=list(labels_scores_dict.keys()),
        cols_labels=[col_label],
        name=title
    )

    if rows_unlabeled:
        if type_k:
            rows_unlabeled = rows_unlabeled[col_label]

        input_mat.row_bind(
            matrix=Matrix(
                mat=np.transpose(np.array([list(rows_unlabeled.values())])),
                rows_labels=list(rows_unlabeled.keys()),
                cols_labels=[col_label],
                name=title
            )
        )

    return input_mat.match_delete_rows(kernel.rows_labels).match_missing_rows(kernel.rows_labels,
                                                                              missing_value).match_rows(kernel)


def format_input_matrix_from_type_label_score_dict(
        type_dict_labels_scores_dict: Union[Dict[str, Dict[str, int]], Dict[str, int]],
        kernel,
        missing_value: int = -1,
        rows_unlabeled=None,  # TODO: To discuss
        title=''
) -> Matrix:
    """Generate input matrix from labels scores dict and/or handle type classification by columns."""
    if _type_dict_label_scores_dict_data_struct_check(type_dict_labels_scores_dict):

        init_k = get_random_key_from_dict(type_dict_labels_scores_dict)
        init_v = type_dict_labels_scores_dict.pop(init_k)

        input_mat = format_input_vector_from_label_score_dict(init_v,
                                                              kernel,
                                                              init_k,
                                                              missing_value,
                                                              rows_unlabeled,
                                                              True,
                                                              title=title
                                                              )

        for node_type, scores_dict in type_dict_labels_scores_dict.items():
            input_vector = format_input_vector_from_label_score_dict(scores_dict,
                                                                     kernel,
                                                                     node_type,
                                                                     missing_value,
                                                                     rows_unlabeled,
                                                                     True
                                                                     )
            input_mat.col_bind(matrix=input_vector)

        return input_mat
    else:
        return format_input_vector_from_label_score_dict(type_dict_labels_scores_dict, kernel)

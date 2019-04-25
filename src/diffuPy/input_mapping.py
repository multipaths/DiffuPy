import itertools

from diffuPy.matrix import Matrix
from diffuPy.utils import check_substrings, get_labels_set_from_dict


def generate_categoric_input_from_labels(rows_labels, cols_labels, background_mat):
    input_mat = Matrix(rows_labels=rows_labels, cols_labels=cols_labels, init=1)
    return input_mat.match_missing_rows(background_mat.rows_labels, 0).match_rows(background_mat)


def get_mapping(to_map, background_map, mirnas=None, submapping=None, title='', print_percentage=False):

    intersection = to_map.intersection(background_map)

    if mirnas:
        mirnas_substring = [e for e in check_substrings(mirnas['micrornas'], background_map) if 'mir' in e]
        intersection = intersection.union(mirnas_substring)

    if submapping:
        intersection = intersection.intersection(submapping)

    if len(intersection) != 0 and print_percentage:
        print(f'{title} ({str(len(intersection))}) {(len(intersection) / len(to_map)) * 100}%')

    return intersection


def get_mapping_subsets(subsets_dict,
                        map_labels,
                        title,
                        percentage_reference_labels=False,
                        submapping=None,
                        ):
    mapping_dict = {}
    total_dimention = 0

    print('Mapping by ' + title + ':')

    if not isinstance(map_labels, set):
        raise Exception('map_labels must be a set.')

    for type_name, entites in subsets_dict.items():
        percentage = 0

        # TODO: Mapping substring not other than mirnas

        if title == 'entity type/omic' and type_name in ['micrornas', 'mirna', 'mirna_nodes']:
            mirnas = subsets_dict
        else:
            mirnas = None

        mapping = get_mapping(entites,
                              map_labels,
                              mirnas=mirnas,
                              submapping=submapping)

        if len(entites) != 0:
            if percentage_reference_labels:
                percentage = len(mapping) / len(map_labels)
            else:
                percentage = len(mapping) / len(entites)

        print(f'{type_name} ({str(len(mapping))}) {percentage * 100}%')

        mapping_dict[type_name] = (mapping, percentage)

        total_dimention += len(mapping)

    if percentage_reference_labels:
        percentage = total_dimention / len(map_labels)
    else:
        percentage = total_dimention / len(get_labels_set_from_dict(subsets_dict))

    print(f'Total ({total_dimention}) {percentage * 100}% \n')

    return mapping_dict, percentage


def get_mapping_two_dim_subsets(two_dimentional_dict,
                                map_labels,
                                background_labels):
    mapping_dict = {}

    for key, subsets_dict in two_dimentional_dict.items():
        mapping, percentage = get_mapping_subsets(subsets_dict,
                                                  map_labels,
                                                  percentage_reference_labels=True,
                                                  submapping=background_labels,
                                                  title=key.capitalize())

        mapping_dict[key] = (mapping, percentage)

    return mapping_dict

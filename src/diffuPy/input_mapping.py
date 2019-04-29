import itertools

from diffuPy.matrix import Matrix
from diffuPy.utils import check_substrings, get_labels_set_from_dict


def generate_categoric_input_from_labels(rows_labels, cols_labels, background_mat):
    input_mat = Matrix(rows_labels=rows_labels, cols_labels=cols_labels, init=1)
    return input_mat.match_missing_rows(background_mat.rows_labels, 0).match_rows(background_mat)


def get_mapping(to_map,
                background_map,
                mirnas=None,
                mirnas_mapping=None,
                submapping=None,
                title='',
                print_percentage=False):
    intersection = to_map.intersection(background_map)

    if mirnas:
        mirnas_substring = [e for e in check_substrings(mirnas, background_map) if 'mir' in e]
        intersection = intersection.union(mirnas_substring)

    if mirnas_mapping:
        intersection = intersection.union(mirnas_mapping.intersection(to_map))

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
                        mirnas=None,
                        mirnas_mapping=None,
                        relative_statistics=None
                        ):
    entity_type_map = {'metabolite_nodes': 'metabolite', 'mirna_nodes': 'micrornas', 'gene_nodes': 'genes',
                       'bp_nodes': 'bps'}
    mapping_dict = {}
    total_entites = set()

    print('Mapping by ' + title + ':')

    if not isinstance(map_labels, set) and not isinstance(map_labels, list):
        raise Exception('map_labels must be a set or a list.')

    for type_name, entites in subsets_dict.items():

        # TODO: Mapping substring not other than mirnas

        mapping = get_mapping(entites,
                              map_labels,
                              mirnas=mirnas,
                              mirnas_mapping=mirnas_mapping,
                              submapping=submapping)

        percentage = 0

        if percentage_reference_labels:
            percentage = len(mapping) / len(map_labels)
        elif relative_statistics:
            subset_len = len(relative_statistics[entity_type_map[type_name]])
            if subset_len != 0:
                percentage = len(mapping) / subset_len
        else:
            if len(entites) != 0:
                percentage = len(mapping) / len(entites)

        print(f'{type_name} ({str(len(mapping))}) {percentage * 100}%')

        mapping_dict[type_name] = (mapping, percentage)

        total_entites.update(mapping)

    total_dimention = len(total_entites)

    if percentage_reference_labels:
        percentage = total_dimention / len(map_labels)

    else:
        percentage = total_dimention / len(get_labels_set_from_dict(subsets_dict))

    print(f'Total ({total_dimention}) {percentage * 100}% \n')

    return mapping_dict, percentage, total_entites


def get_mapping_two_dim_subsets(two_dimentional_dict,
                                map_labels,
                                background_labels=None,
                                percentage_reference_background_labels=False,
                                mirnas=None,
                                relative_statistics=None,
                                mirnas_mapping=None):
    mapping_dict = {}
    total_entites = set()

    for subset_title, subsets_dict in two_dimentional_dict.items():
        mapping, percentage, entites = get_mapping_subsets(subsets_dict,
                                                           map_labels,
                                                           relative_statistics=relative_statistics,
                                                           submapping=background_labels,
                                                           mirnas=mirnas,
                                                           mirnas_mapping=mirnas_mapping,
                                                           title=subset_title.capitalize()
                                                           )

        mapping_dict[subset_title] = (mapping, percentage)
        total_entites.update(entites)

    total_dimention = len(total_entites)

    if percentage_reference_background_labels:
        total_percentage = total_dimention / len(background_labels)
    else:
        total_percentage = total_dimention / len(map_labels)

    print(f'Total ({total_dimention}) {total_percentage * 100}% \n')

    return mapping_dict, total_percentage, total_dimention

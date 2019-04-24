from diffuPy.matrix import Matrix
from diffuPy.utils import check_substrings


def generate_categoric_input_from_labels(rows_labels, cols_labels, background_mat):
    input_mat = Matrix(rows_labels=rows_labels, cols_labels=cols_labels, init=1)
    return input_mat.match_missing_rows(background_mat.rows_labels, 0).match_rows(background_mat)


def get_mapping(to_map, background_map, mirnas = None, submapping = None):
    #TODO: Mapping substring not other than mirnas

    intersection = to_map.intersection(background_map)

    if mirnas:
        mirnas_substring = [e for e in check_substrings(mirnas['micrornas'], background_map)if 'mir' in e]
        return intersection.union(mirnas_substring)

    if submapping:
        return intersection.intersection(submapping)

    return intersection

from itertools import combinations
from typing import Tuple, List
from numpy.typing import ArrayLike

import numpy as np

vectorized_len = np.vectorize(len)


def first_element(a):
    return a[0]
return_first_elements = np.vectorize(first_element)

def second_element(a):
    return a[1]
return_second_elements = np.vectorize(second_element)


def get_support(data):
    """Get support

    Parameters
    ----------
    data: List[set] :

    Returns
    -------

    """
    # return [len(value) for value in data]
    return vectorized_len(data)


def frequent_single_itemsets(
    data, min_support: int = 0
):
    """

    Parameters
    ----------
    data: Data :

    min_support: int :
         (Default value = 0)

    Returns
    -------
    type


    """
    columns_support = get_support(data)
    support_mask = columns_support > min_support
    # Reduce by support
    data = data[support_mask]
    support = columns_support[support_mask]

    def create_sets(a):
        return {a}

    transform_to_sets = np.vectorize(create_sets)

    if len(np.arange(len(support_mask))[support_mask]) > 0:
        candidates = transform_to_sets(np.arange(len(support_mask))[support_mask])
    else:
        candidates = np.array[{}]
    return candidates, support, data


def generate_candidates(
    previous_candidates,
    previous_multiplier_mask=None
):
    """Generate candidate set from `previous_candidates` with size `k`

    Parameters
    ----------
    previous_candidates: List[set] :

    previous_multiplier_mask: List[int] :
         (Default value = None)

    Returns
    -------

    """
    # number of previous candidates
    d = len(previous_candidates)

    # If no previous candidates then return empty arrays
    if d <= 1:
        return np.array([])

    if previous_multiplier_mask is None:  # Generate candidates with size 2
        perm_all = list(combinations(range(0, d), 2))
    else:  # Generate candidates with size > 2
        previous_multiplier_mask_left = return_first_elements(previous_multiplier_mask)
        count_arr = np.bincount(previous_multiplier_mask_left)
        no_zeros_mask = count_arr > 0
        count_arr = count_arr[no_zeros_mask]
        cum_n = np.cumsum(count_arr)
        cum_n = np.concatenate((np.array([0]), cum_n))
        perm_all = []
        if len(count_arr) == 0:
            return np.array([])
        for i, count in enumerate(count_arr):
            if count == 1:
                continue
            offset = cum_n[i]
            perm = list(combinations(range(offset, offset + count), 2))
            perm_all = perm_all + perm
    out = np.empty(len(perm_all), dtype=object)
    out[:] = perm_all
    return out


def itemsets_support(
    data, multiplier_mask
):
    """Get support for `itemsets` and return sets with minimal `support

    Parameters
    ----------
    data: Data :

    multiplier_mask: List[tuple] :


    Returns
    -------

    """
    #data = np.array([data[left].intersection(data[right]) for left, right in multiplier_mask])
    def intersection(a):
        return data[a[0]].intersection(data[a[1]])

    intersection_vect = np.vectorize(intersection)

    if len(multiplier_mask) > 0:
        data = intersection_vect(multiplier_mask)
    else:
        data = np.array[{}]

    data_support = get_support(data)

    return data, data_support


def min_support_set(
    previous_candidates,
    candidates_support,
    data,
    multiplier_mask,
    min_support: int = 0,
):
    """

    Parameters
    ----------
    previous_candidates: List[set] :

    candidates_support: List[int] :

    data: List[set] :

    multiplier_mask: List[tuple] :

    min_support: int :
         (Default value = 0)

    Returns
    -------
    type

    """
    # Reduce by support
    mask = candidates_support >= min_support

    candidates_support = candidates_support[mask]
    data = data[mask]
    multiplier_mask = multiplier_mask[mask]

    def plus(a):
        return previous_candidates[a[0]].union(previous_candidates[a[1]])
    plus_vect = np.vectorize(plus)
    if len(multiplier_mask) > 0:
        candidates = plus_vect(multiplier_mask)
    else:
        candidates = np.array({})
    return candidates, candidates_support, multiplier_mask, data

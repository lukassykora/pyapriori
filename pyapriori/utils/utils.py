from itertools import combinations
from typing import Tuple, List

import numpy as np


def get_support(data: List[set]) -> List[int]:
    """Get support

    Parameters
    ----------
    data: List[set] :

    Returns
    -------

    """
    return [len(value) for value in data]


def frequent_single_itemsets(
    data: List[set], min_support: int = 0
) -> Tuple[List[set], List[int], List[set]]:
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
    # Reduce by support
    data = [data[i] for i, support in enumerate(columns_support) if
            support >= min_support]
    support = [support for support in columns_support if support >= min_support]
    candidates = [set(i) for i, support in enumerate(columns_support) if
                  support >= min_support]
    return candidates, support, data


def generate_candidates(
    previous_candidates: List[set], previous_multiplier_mask: List[tuple] = None
) -> List[tuple]:
    """Generate candidate set from `previous_candidates` with size `k`

    Parameters
    ----------
    previous_candidates: np.ndarray :

    previous_multiplier_mask: List[int] :
         (Default value = None)

    Returns
    -------

    """
    if previous_multiplier_mask is not None:
        previous_multiplier_mask = [a for a, b in previous_multiplier_mask]

    # number of previous candidates
    d = len(previous_candidates)

    # If no previous candidates then return empty arrays
    if d <= 1:
        return []

    if previous_multiplier_mask is None:  # Generate candidates with size 2
        perm_all = list(combinations(range(0, d), 2))
    else:  # Generate candidates with size > 2
        count_arr = np.bincount(previous_multiplier_mask)
        no_zeros_mask = count_arr > 0
        count_arr = count_arr[no_zeros_mask]
        cum_n = np.cumsum(count_arr)
        cum_n = np.concatenate((np.array([0]), cum_n))
        perm_all = []
        if len(count_arr) == 0:
            return []
        for i, count in enumerate(count_arr):
            if count == 1:
                continue
            offset = cum_n[i]
            perm = list(combinations(range(offset, offset + count), 2))
            perm_all = perm_all + perm
    return perm_all


def itemsets_support(
    data: List[set], multiplier_mask: List[tuple]
) -> Tuple[List[set], List[int]]:
    """Get support for `itemsets` and return sets with minimal `support

    Parameters
    ----------
    data: Data :

    multiplier_mask: List[tuple] :


    Returns
    -------

    """
    data = [data[left].intersection(data[right]) for left, right in multiplier_mask]
    data_support = get_support(data)
    return data, data_support


def min_support_set(
    previous_candidates: List[set],
    candidates_support: List[int],
    data: List[set],
    multiplier_mask: List[tuple],
    min_support: int = 0,
) -> Tuple[List[set], List[int], List[tuple], Data]:
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
    mask = [i for i, support in enumerate(candidates_support) if
            support >= min_support]

    candidates = [candidate for i, candidate in enumerate(previous_candidates) if
                  i in mask]
    candidates_support = [support for i, support in enumerate(candidates_support) if
                          i in mask]
    data = [value for i, value in enumerate(data) if
            i in mask]
    multiplier_mask = [value for i, value in enumerate(multiplier_mask) if
                       i in mask]

    return candidates, candidates_support, multiplier_mask, data

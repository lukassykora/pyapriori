from itertools import combinations
from typing import Union, Tuple, List, Set

import cupy as cp
import numpy as np

Data = dict[int, Set[int]]
Support = dict[int, int]


def get_support(data: Data) -> Support:
    """Get support

    Parameters
    ----------
    data: Data :

    Returns
    -------

    """
    return {i: len(value) for i, value in data.items()}


def frequent_single_itemsets(
    data: Data, min_support: int = 0
) -> Tuple[Support, Data]:
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
    data = {feature_key: data[feature_key] for feature_key, support in
            columns_support.items() if support >= min_support}
    support = {i: support for i, support in columns_support.items() if i in data.keys()}
    return support, data


def generate_candidates(
    previous_candidates: List[int],
    previous_multiplier_mask: List[int] = None,
) -> List[Tuple[int, int]]:
    """Generate candidate set from `previous_candidates` with size `k`

    Parameters
    ----------
    previous_candidates: np.ndarray :

    previous_multiplier_mask: List[int] :
         (Default value = None)

    Returns
    -------

    """

    # number of previous candidates
    d = len(previous_candidates)

    # If no previous candidates then return empty arrays
    if d <= 1:
        return []

    if previous_multiplier_mask is None:  # Generate candidates with size 2
        multiplier_mask = combinations(range(0, d), 2)
    else:  # Generate candidates with size > 2
        count_arr = np.bincount([left for left, right in previous_multiplier_mask])
        count_arr = count_arr[count_arr > 0]
        cum_n = np.cumsum(count_arr)
        cum_n = np.concatenate((np.array([0]), cum_n))
        cum_n = cum_n[:-1].copy()
        multiplier_mask = []
        if len(count_arr) == 0:
            return []
        for i, count in enumerate(count_arr):
            if count == 1:
                continue
            offset = cum_n[i]
            multiplier_mask += combinations(range(offset, offset + count), 2)

    return multiplier_mask


def itemsets_support(data: Data, multiplier_mask: List[Tuple[int, int]]) -> Tuple[Data, MultiDimensionalArray]:
    """Get support for `itemsets` and return sets with minimal `support

    Parameters
    ----------
    data: Data :

    multiplier_mask_left: List[int] :

    multiplier_mask_right: List[int] :


    Returns
    -------

    """
    data = {i: data[left].intersection(data[right]) for i, (left, right) in
            enumerate(multiplier_mask)}
    data_support = get_support(data)
    return data, data_support


def min_support_set(
    previous_candidates: np.ndarray,
    candidates_support: MultiDimensionalArray,
    data: Data,
    multiplier_mask_left: List[int],
    multiplier_mask_right: List[int],
    min_support: int = 0,
) -> Tuple[MultiDimensionalArray, MultiDimensionalArray, List[int], Data]:
    """

    Parameters
    ----------
    previous_candidates: np.ndarray :

    candidates_support:MultiDimensionalArray :

    data: Data :

    multiplier_mask_left:List[int] :

    multiplier_mask_right:List[int] :

    min_support: int :
         (Default value = 0)

    Returns
    -------
    type


    """
    # Reduce by support
    data = {feature_key: data[feature_key] for feature_key, support in
            enumerate(candidates_support) if support >= min_support}
    candidates_support = [support for support in candidates_support if support >= min_support]


    return candidates, candidates_support, multiplier_mask_left, data

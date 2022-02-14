from itertools import combinations
from typing import Union, Tuple, List, Set

import cupy as cp
import numpy as np

Data = dict[int, Set[int]]
MultiDimensionalArray = Union[np.ndarray, cp.ndarray]


def get_support(data: Data) -> List[int]:
    """Get support

    Parameters
    ----------
    data: Data :

    Returns
    -------

    """
    return [len(value) for value in data.values()]


def frequent_single_itemsets(
    data: Data, min_support: int = 0
) -> Tuple[MultiDimensionalArray, MultiDimensionalArray, Data]:
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
            enumerate(columns_support) if support >= min_support}
    support = [support for support in columns_support if support >= min_support]
    return support, data


def generate_candidates(
    previous_candidates: List[int],
    previous_multiplier_mask: List[int] = None,
) -> Tuple[List[int], List[int]]:
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
        return [], []

    if previous_multiplier_mask is None:  # Generate candidates with size 2
        perm = combinations(range(0, d), 2)
        multiplier_mask_left, multiplier_mask_right = map(list, zip(*perm))
    else:  # Generate candidates with size > 2
        count_arr = np.bincount(previous_multiplier_mask)
        no_zeros_mask = count_arr > 0
        count_arr = count_arr[no_zeros_mask]
        cum_n = np.cumsum(count_arr)
        cum_n = np.concatenate((np.array([0]), cum_n))
        cum_n = cum_n[:-1].copy()
        multiplier_mask_left = []
        multiplier_mask_right = []
        if len(count_arr) == 0:
            return [], []
        for i, count in enumerate(count_arr):
            if count == 1:
                continue
            offset = cum_n[i]
            perm = combinations(range(offset, offset + count), 2)
            multiplier_mask_left_part, multiplier_mask_right_part = map(
                list, zip(*perm)
            )
            multiplier_mask_left += multiplier_mask_left_part
            multiplier_mask_right += multiplier_mask_right_part

    return multiplier_mask_left, multiplier_mask_right


def itemsets_support(
    data: Data, multiplier_mask_left: List[int], multiplier_mask_right: List[int]
) -> Tuple[Data, MultiDimensionalArray]:
    """Get support for `itemsets` and return sets with minimal `support

    Parameters
    ----------
    data: Data :

    multiplier_mask_left: List[int] :

    multiplier_mask_right: List[int] :


    Returns
    -------

    """
    data = {i: data[left].intersection(data[multiplier_mask_right[i]]) for i, left in
            enumerate(multiplier_mask_left)}
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

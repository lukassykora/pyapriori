from types import ModuleType
from itertools import combinations
from typing import Union, Tuple, List

import cupy as cp
import numpy as np
from cupyx.scipy.sparse import csr_matrix as cupy_csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix

Data = Union[np.ndarray, cp.ndarray, csr_matrix, csc_matrix, cupy_csr_matrix]
MultiDimensionalArray = Union[np.ndarray, cp.ndarray]


def get_support(data: Data, numpy_or_cupy: ModuleType) -> MultiDimensionalArray:
    """Get support

    Parameters
    ----------
    data: Data :

    numpy_or_cupy: types.ModuleType('numpy' :

    'cupy') :


    Returns
    -------

    """
    if isinstance(data, (csr_matrix, csc_matrix, cupy_csr_matrix)):
        return numpy_or_cupy.array(data.sum(axis=0)).ravel()
    return data.sum(axis=0)


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
    numpy_or_cupy = get_numpy_or_cupy(data)

    if isinstance(data, (cupy_csr_matrix)):
        data = data.astype(numpy_or_cupy.float32)
    else:
        data = data.astype(numpy_or_cupy.bool_)

    columns_support = get_support(data, numpy_or_cupy)

    # Reduce by support
    indices = numpy_or_cupy.arange(max(columns_support.shape))
    over_support_mask = columns_support >= min_support
    reduced_indices = indices[over_support_mask]
    reduced_columns_support = columns_support[over_support_mask]

    # Sort it by support
    support_sorted_mask = numpy_or_cupy.argsort(reduced_columns_support)
    reduced_indices_sorted = reduced_indices[support_sorted_mask]
    reduced_columns_support_sorted = reduced_columns_support[support_sorted_mask]

    # Indices Matrix
    matrix_size = len(over_support_mask)
    indices_matrix = numpy_or_cupy.zeros((matrix_size, matrix_size))
    numpy_or_cupy.fill_diagonal(indices_matrix, 1)
    indices_matrix = indices_matrix[reduced_indices_sorted].astype(bool)

    # Reduce data
    data = data[:, reduced_indices_sorted]

    return indices_matrix, reduced_columns_support_sorted, data


def get_numpy_or_cupy(data: Data) -> ModuleType:
    """

    Parameters
    ----------
    data: Data) -> types.ModuleType('numpy' :

    'cupy' :


    Returns
    -------

    """
    if isinstance(data, (cupy_csr_matrix, cp.ndarray)):
        return cp
    return np


def generate_candidates(
    previous_candidates: MultiDimensionalArray,
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
    d = previous_candidates.shape[0]

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
    numpy_or_cupy = get_numpy_or_cupy(data)
    if numpy_or_cupy == cp:
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()

    if isinstance(data, (csr_matrix, csc_matrix, cupy_csr_matrix)):
        data = data[:, multiplier_mask_left].multiply(data[:, multiplier_mask_right])
    else:
        data = data[:, multiplier_mask_left] * data[:, multiplier_mask_right]

    data_support = get_support(data, numpy_or_cupy)
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
    numpy_or_cupy = get_numpy_or_cupy(data)

    over_support_mask = candidates_support >= min_support

    data = data[:, over_support_mask]
    candidates_support = candidates_support[over_support_mask]
    multiplier_mask_left = numpy_or_cupy.array(multiplier_mask_left)[
        over_support_mask
    ].tolist()
    multiplier_mask_right = numpy_or_cupy.array(multiplier_mask_right)[
        over_support_mask
    ].tolist()

    candidates = (
        previous_candidates[multiplier_mask_left, :]
        + previous_candidates[multiplier_mask_right, :]
    )
    return candidates, candidates_support, multiplier_mask_left, data

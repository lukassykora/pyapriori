from typing import Union, Tuple, List
import types
from itertools import combinations
import numpy as np
import cupy as cp
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from cupyx.scipy.sparse import csr_matrix as cupy_csr_matrix

Data = Union[np.ndarray, cp.ndarray, csr_matrix, csc_matrix, cupy_csr_matrix]
MultiDimensionalArray = Union[np.ndarray, cp.ndarray]


def get_support(data: Data, numpy_or_cupy: types.ModuleType('numpy', 'cupy')) -> MultiDimensionalArray:
    if isinstance(data, (csr_matrix, csc_matrix, cupy_csr_matrix)):
        return numpy_or_cupy.array(data.sum(axis=0)).ravel()
    return data.sum(axis=0)


def frequent_single_itemsets(data: Data, min_support: int = 0) -> Tuple[np.ndarray, MultiDimensionalArray, Data]:
    """Return one-item itemsets with at least `support` support."""
    numpy_or_cupy = get_numpy_or_cupy(data)

    if isinstance(data, cupy_csr_matrix):
        data = data.astype(float)

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

    # Reduce data
    data = data[:, reduced_indices_sorted]

    # Return sets
    reduced_indices_sorted_sets = np.array([{int(i)} for i in reduced_indices_sorted])

    return reduced_indices_sorted_sets, reduced_columns_support_sorted, data


def get_numpy_or_cupy(data: Data) -> types.ModuleType('numpy', 'cupy'):
    if isinstance(data, (cupy_csr_matrix, cp.ndarray)):
        return cp
    return np

def generate_candidates(previous_candidates: np.ndarray, k: int, previous_multiplier_mask: List[int] = None) -> Tuple[np.ndarray, List[int], List[int]]:
    """Generate candidate set from `previous_candidates` with size `k`"""

    # number of previous candidates
    d = len(previous_candidates)

    # If no previous candidates then return empty arrays
    if d<=1:
        return np.array([]), [], []

    if previous_multiplier_mask is None: # Generate candidates with size 2
        perm = combinations(range(0, d), 2)
        multiplier_mask_left, multiplier_mask_right = map(list,zip(*perm))
    else: # Generate candidates with size > 2
        count_arr = np.bincount(previous_multiplier_mask)
        no_zeros_mask = count_arr > 0
        count_arr = count_arr[no_zeros_mask]
        cum_n = np.cumsum(count_arr)
        cum_n = np.concatenate((np.array([0]), cum_n))
        cum_n = cum_n[:-1].copy()
        multiplier_mask_left = []
        multiplier_mask_right = []
        if len(count_arr)==0:
            return np.array([]), [], []
        for i, count in enumerate(count_arr):
            if count == 1:
                continue
            offset = cum_n[i]
            perm = combinations(range(offset, offset + count), 2)
            multiplier_mask_left_part, multiplier_mask_right_part = map(list,zip(*perm))
            multiplier_mask_left += multiplier_mask_left_part
            multiplier_mask_right += multiplier_mask_right_part

    if len(multiplier_mask_left) == 0:
        return np.array([]), [], []
    else: # Generate new candidates by union
        candidates = previous_candidates[multiplier_mask_left] | previous_candidates[multiplier_mask_right]

    return candidates, multiplier_mask_left, multiplier_mask_right

def itemsets_support(data: Data, multiplier_mask_left: List[int], multiplier_mask_right: List[int]) -> Tuple[Data, MultiDimensionalArray]:
    """Get support for `itemsets` and return sets with minimal `support"""

    if isinstance(data, (csr_matrix, csc_matrix, cupy_csr_matrix)):
        data = data[:, multiplier_mask_left].multiply(data[:, multiplier_mask_right])
    else:
        data = data[:, multiplier_mask_left] * data[:, multiplier_mask_right]

    data_support = get_support(data, get_numpy_or_cupy(data))
    return data, data_support


def min_support_set(candidates: np.ndarray, candidates_support:MultiDimensionalArray, data: Data, multiplier_mask_left:List[int], min_support: int = 0)-> Tuple[np.ndarray, MultiDimensionalArray, List[int], Data]:
    """Return sets with minimal `support`"""
    numpy_or_cupy = get_numpy_or_cupy(data)

    over_support_mask = candidates_support >= min_support

    data = data[:,over_support_mask]
    candidates_support = candidates_support[over_support_mask]
    if isinstance(over_support_mask, cp.ndarray):
        candidates = candidates[cp.asnumpy(over_support_mask)]
    else:
        candidates = candidates[over_support_mask]
    multiplier_mask_left = numpy_or_cupy.array(multiplier_mask_left)[over_support_mask].tolist()

    return candidates, candidates_support, multiplier_mask_left, data

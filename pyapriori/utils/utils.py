from typing import Tuple, List
from numpy.typing import ArrayLike
from types import ModuleType
from cupyx.scipy.sparse import csr_matrix as cupy_csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from numba import njit

import numpy as np

Data = Union[np.ndarray, cp.ndarray, csr_matrix, csc_matrix, cupy_csr_matrix]


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
    elif isinstance(data, (np.ndarray, csr_matrix, csc_matrix)):
        return np
    else:
        raise TypeError("Only np.ndarray, cp.ndarray, csr_matrix, csc_matrix, cupy_csr_matrix are allowed")


def bitwise_support_all_columns(data: Data, m: int):
    n = data.shape[0]  # number of rows
    result = np.zeros(m, dtype=np.int32)  # empty results array
    for i in range(0, n, 4):
        for j in range(0, m, 8):
            if i + 3 < n and j + 7 < m:  # Optimized 4x8 tile computation
                k = j // 8
                b0, b1, b2, b3 = packed[i, k], packed[i + 1, k], packed[i + 2, k], packed[i + 3, k]
                for j2 in range(8):
                    shift = 7 - j2
                    mask = 1 << shift
                    result[j + j2] += ((b0 & mask) + (b1 & mask) + (b2 & mask) + (b3 & mask)) >> shift
            else:  # Slow fallback computation
                for i2 in range(i, min(i + 4, n)):
                    for j2 in range(j, min(j + 8, m)):
                        result[j2] += bool(packed[i2, j2 // 8] & (128 >> (j2 % 8)))
    return result


def support_all_columns(data: Data):
    return data.sum(axis=0)


def slice_array_columns(array, columns):
    return data[:, columns]


def frequent_single_itemsets(data: Data, numpy_or_cupy: ModulType, number_of_columns: int, min_support: int = 0,
                             is_ordered: bool = True, is_bitwise: bool = False, is_numba: bool = False):
    """

    Parameters
    ----------
    data: Data :

    numpy_or_cupy: ModulType :

    number_of_columns: int :

    min_support: int :
         (Default value = 0)

    is_ordered: bool :
         (Default value = True)

    is_bitwise: bool :
         (Default value = False)

    is_numba: bool :
         (Default value = False)

    Returns
    -------
    type


    """

    if is_bitwise:
        if is_numba:
            support = njit()(bitwise_support_all_columns)(data, number_of_columns)
        else:
            support = bitwise_support_all_columns(data, number_of_columns)
    else:
        if is_numba:
            support = njit()(support_all_columns)(data)
        else:
            support = support_all_columns(data)
    if is_ordered:
        support_order = numpy_or_cupy.argsort(support)
        support_sorted = support[support_order]
    else:
        support_order = numpy_or_cupy.arange(number_of_columns)
        support_sorted = support
    min_support_mask = support_sorted >= min_support
    features_list = support_order[min_support_mask]
    features_support = support_sorted[min_support_mask]
    if not is_bitwise:
        if is_numba:
            transactions = njit()(slice_array_columns)(data, features_list)
        else:
            transactions = slice_array_columns(data, features_list)
    else:
        transactions = None

    return transactions, features_list, features_support


def add_candidates(prefix_list, candidates_list, transactions_list, prefix, candidates, transactions,
                   min_length: int = 1):
    if len(prefix) > min_length or len(candidates) < 2:
        return
    prefix_list.append(prefix)
    candidates_list.append(candidates)
    transactions_list.append(transactions)


def add_result(frequent_itemsets, new_prefix, new_candidates, candidates_support):
    for j, row in enumerate(new_candidates):
        frequent_itemsets.append((new_prefix + [row], candidates_support[j]))


def support_sliced_columns(array, i, reduced_candidates):
    mask = array[:, i]  # 15
    new_array = array[:, reduced_candidates][mask, :]
    array_support = new_array.sum(axis=0)
    return array_support, new_array


def bitwise_support_sliced_columns(array, i, reduced_candidates):
    n = packed.shape[0]
    res = np.zeros(len(cols), dtype=np.int32)
    byte = i//8
    bit = 128>>(i%8)
    for i in range(n):
        if packed[i, byte] & bit:
            for ji,j in enumerate(cols):
                res[ji] += bool(packed[i, j//8] & (128>>(j%8)))
    return res


def frequent_sliced_itemsets(array, i, reduced_candidates, data):
    if is_bitwise:
        if is_numba:
            support = njit()(bitwise_support_sliced_columns)(data, i, reduced_candidates)
        else:
            support = bitwise_support_sliced_columns(data, i, reduced_candidates)
        return support, None
    else:
        if is_numba:
            support, new_array = njit()(support_sliced_columns)(array, i, reduced_candidates)
        else:
            support, new_array = support_sliced_columns(array, i, reduced_candidates)
        return support, new_array


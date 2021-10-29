from typing import Union, Tuple
import types
import numpy as np
import cupy as cp
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from cupyx.scipy.sparse import csr_matrix as cupy_csr_matrix

Data = Union[np.ndarray, cp.ndarray, csr_matrix, csc_matrix, cupy_csr_matrix]
MultiDimensionalArray = Union[np.ndarray, cp.ndarray]
ItemsetsSupport = Tuple[MultiDimensionalArray, MultiDimensionalArray, Data]


def frequent_single_itemsets(data: Data, min_support: int = 0) -> ItemsetsSupport:
    """Return one-item itemsets with at least `support` support."""
    numpy_or_cupy = get_numpy_or_cupy(data)

    if isinstance(data, cupy_csr_matrix):
        data = data.astype(float)
        columns_support = numpy_or_cupy.array(data.sum(axis=0)).ravel()
    elif isinstance(data, (csr_matrix, csc_matrix)):
        columns_support = numpy_or_cupy.array(data.sum(axis=0)).ravel()
    else:
        columns_support = data.sum(axis=0)

    # Reduce by support
    indices = numpy_or_cupy.arange(max(columns_support.shape))
    over_support_mask = columns_support >= min_support
    reduced_indices = indices[over_support_mask]
    reduced_columns_support = columns_support[over_support_mask]

    # Sort it by support
    support_sorted_mask = numpy_or_cupy.argsort(reduced_columns_support)
    reduced_indices_sorted = reduced_indices[support_sorted_mask]
    reduced_columns_support_sorted = reduced_columns_support[support_sorted_mask]

    data = data[:, reduced_indices_sorted]

    return reduced_indices_sorted, reduced_columns_support_sorted, data


def get_numpy_or_cupy(data: Data) -> types.ModuleType('numpy', 'cupy'):
    if isinstance(data, (cupy_csr_matrix, cp.ndarray)):
        return cp
    return np

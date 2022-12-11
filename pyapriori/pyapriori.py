"""Main module."""
import time
import numpy as np
import cupy as cp
from numpy.typing import ArrayLike
from pyapriori.utils.utils import (
    frequent_single_itemsets,
    get_numpy_or_cupy,
    frequent_sliced_itemsets,
    add_candidates,
    add_result,
)


class PyApriori:
    """ """

    def __init__(self, min_support: int = 2, min_length: int = 1, max_length: int = None):
        self.min_support = min_support
        self.min_length = min_length
        self.max_length = max_length

    def fit(self, data, **kwargs):
        """

        Parameters
        ----------
        data: Data :


        Returns
        -------

        """
        numpy_or_cupy = get_numpy_or_cupy(data)

        if 'is_bitwise' in kwargs:
            is_bitwise = kwargs['is_bitwise']
            number_of_columns = kwargs['number_of_columns']
        else:
            is_bitwise = False
            number_of_columns = data.shape[1]

        if 'is_ordered' in kwargs:
            is_ordered = kwargs['is_ordered']
        else:
            is_ordered = True

        if 'is_numba' in kwargs:
            is_numba = kwargs['is_numba']
        else:
            is_numba = True

        transactions, candidates, candidates_support = frequent_single_itemsets(
            data, numpy_or_cupy, number_of_columns, self.min_support, is_ordered, is_bitwise, is_numba
        )

        prefix_list = []
        candidates_list = []
        transactions_list = []

        frequent_itemsets = []
        stop_dict = {}

        add_candidates(prefix_list, candidates_list, transactions_list, [], candidates, transactions,
                       self.max_length)

        add_result(frequent_itemsets, [], candidates, candidates_support, self.min_length)

        while len(transactions_list) > 0:
            prev_prefix = prefix_list.pop(0)
            prev_candidates = candidates_list.pop(0)
            prev_transactions = transactions_list.pop(0)

            for i, element in enumerate(prev_candidates[:-1]):
                new_prefix = prev_prefix + [element]
                reduced_candidates = prev_candidates[i + 1:]

                # STOP
                if tuple(new_prefix[1:]) in stop_dict:
                    prev_stop = stop_dict[tuple(new_prefix[1:])]
                    _, interse, _ = numpy_or_cupy.intersect1d(reduced_candidates, prev_stop, assume_unique=True,
                                                              return_indices=True)
                    reduced_candidates = numpy_or_cupy.delete(reduced_candidates, interse)
                else:
                    prev_stop = numpy_or_cupy.array([])

                candidates_support, new_transactions = frequent_sliced_itemsets(prev_transactions, i,
                                                                                reduced_candidates, data, is_bitwise,
                                                                                is_numba)

                support_mask = candidates_support >= self.min_support

                # STOP
                new_stop = reduced_candidates[~support_mask]
                stop_dict[tuple(new_prefix)] = np.concatenate((new_stop, prev_stop))

                # Reduce by support
                if not is_bitwise:
                    new_transactions = new_transactions[:, support_mask]
                new_candidates = reduced_candidates[support_mask]
                candidates_support = candidates_support[support_mask]

                add_candidates(prefix_list, candidates_list, transactions_list, new_prefix, new_candidates,
                               new_transactions, self.max_length)

                add_result(frequent_itemsets, new_prefix, new_candidates, candidates_support, self.min_length)
        return frequent_itemsets

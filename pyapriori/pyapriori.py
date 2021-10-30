"""Main module."""
from pyapriori.utils.utils import frequent_single_itemsets, generate_candidates, itemsets_support, min_support_set, Data, get_numpy_or_cupy
import numpy as np

class PyApriori:

    def __init__(self, min_support: int = 2, min_length: int = 2):
        self.min_support = min_support
        self.min_length = min_length

    def fit(self, data: Data) -> tuple:
        numpy_or_cupy = get_numpy_or_cupy(data)
        candidates, candidates_support, data = frequent_single_itemsets(data, self.min_support)
        k = 2
        multiplier_mask_left = None
        result = None
        result_support = None
        if self.min_length < 2:
            result = candidates
            result_support = candidates_support
        while candidates.size > 0:
            candidates, multiplier_mask_left, multiplier_mask_right = generate_candidates(candidates, k, multiplier_mask_left)
            if candidates.size == 0:
                break

            data, candidates_support = itemsets_support(data, multiplier_mask_left, multiplier_mask_right)
            candidates, candidates_support, multiplier_mask_left, data = min_support_set(candidates, candidates_support, data, multiplier_mask_left, self.min_support)

            if k >= self.min_length:
                if result is not None:
                    result = np.concatenate((result, candidates), axis=0)
                    result_support = numpy_or_cupy.concatenate((result_support, candidates_support), axis=0)
                else:
                    result = candidates
                    result_support = candidates_support
            k += 1
        return result, result_support

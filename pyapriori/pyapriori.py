"""Main module."""
import time
import numpy as np
from numpy.typing import ArrayLike
from pyapriori.utils.utils import (
    frequent_single_itemsets,
    generate_candidates,
    itemsets_support,
    min_support_set,
)


class PyApriori:
    """ """

    def __init__(self, min_support: int = 2, min_length: int = 2):
        self.min_support = min_support
        self.min_length = min_length

    def fit(self, data) -> tuple:
        """

        Parameters
        ----------
        data: Data :


        Returns
        -------

        """
        t1 = 0
        t2 = 0
        t3 = 0
        t4 = 0
        start = time.time()
        candidates, candidates_support, data = frequent_single_itemsets(
            data, self.min_support
        )
        end = time.time()
        t1 += end - start
        k = 2
        multiplier_mask = None
        result = np.array([])
        result_support = np.array([])
        if self.min_length < 2:
            result = candidates
            result_support = candidates_support
        while len(candidates) > 0:
            start = time.time()
            multiplier_mask = generate_candidates(
                candidates, multiplier_mask
            )
            end = time.time()
            t2 += end - start
            if len(multiplier_mask) == 0:
                break
            start = time.time()
            data, candidates_support = itemsets_support(
                data, multiplier_mask
            )
            end = time.time()
            t3 += end - start
            start = time.time()
            (
                candidates,
                candidates_support,
                multiplier_mask,
                data,
            ) = min_support_set(
                candidates,
                candidates_support,
                data,
                multiplier_mask,
                self.min_support,
            )
            end = time.time()
            t4 += end - start
            if k >= self.min_length:
                result = np.append(result, candidates)
                result_support = np.append(result_support, candidates_support)
            k += 1
        print('time')
        print(t1)
        print(t2)
        print(t3)
        print(t4)
        return result, result_support

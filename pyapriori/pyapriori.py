"""Main module."""
from typing import List
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

    def fit(self, data: List[set]) -> tuple:
        """

        Parameters
        ----------
        data: Data :


        Returns
        -------

        """
        candidates, candidates_support, data = frequent_single_itemsets(
            data, self.min_support
        )
        k = 2
        multiplier_mask = None
        result = []
        result_support = []
        if self.min_length < 2:
            result = candidates
            result_support = candidates_support
        while len(candidates) > 0:
            multiplier_mask = generate_candidates(
                candidates, multiplier_mask
            )
            if len(multiplier_mask) == 0:
                break
            data, candidates_support = itemsets_support(
                data, multiplier_mask
            )
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
            if k >= self.min_length:
                result += candidates
                result_support += candidates_support
            k += 1
        return result, result_support

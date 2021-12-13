"""Main module."""
from pyapriori.utils.utils import (
    frequent_single_itemsets,
    generate_candidates,
    itemsets_support,
    min_support_set,
    Data,
    get_numpy_or_cupy,
)
import time


class PyApriori:
    """ """

    def __init__(self, min_support: int = 2, min_length: int = 2):
        self.min_support = min_support
        self.min_length = min_length

    def fit(self, data: Data) -> tuple:
        """

        Parameters
        ----------
        data: Data :


        Returns
        -------

        """
        numpy_or_cupy = get_numpy_or_cupy(data)

        candidates, candidates_support, data, counts = frequent_single_itemsets(
            data, self.min_support
        )
        k = 2
        multiplier_mask_left = None
        result = None
        result_support = None
        if self.min_length < 2:
            result = candidates
            result_support = candidates_support
        t1 = 0
        t2 = 0
        t3 = 0
        t4 = 0
        t5 = 0
        tg = 0
        ts = 0
        tr = 0
        te = 0
        while candidates.size > 0:
            start = time.time()
            multiplier_mask_left, multiplier_mask_right, t1, t2, t3 = generate_candidates(
                candidates, multiplier_mask_left, t1, t2, t3
            )
            end = time.time()
            tg += end - start
            if len(multiplier_mask_left) == 0:
                break

            start = time.time()
            data, candidates_support, counts, t4, t5 = itemsets_support(
                data, multiplier_mask_left, multiplier_mask_right, counts, t4, t5
            )
            end = time.time()
            ts += end - start

            start = time.time()
            (
                candidates,
                candidates_support,
                multiplier_mask_left,
                data,
                counts
            ) = min_support_set(
                candidates,
                candidates_support,
                data,
                multiplier_mask_left,
                multiplier_mask_right,
                self.min_support,
                counts
            )
            end = time.time()
            tr += end - start

            start = time.time()
            if k >= self.min_length:
                if result is not None:
                    #result = numpy_or_cupy.vstack((result, candidates))
                    #result_support = numpy_or_cupy.hstack((result_support, candidates_support))
                    result = numpy_or_cupy.concatenate((result, candidates), axis=0)
                    result_support = numpy_or_cupy.concatenate(
                        (result_support, candidates_support), axis=0
                    )
                else:
                    result = candidates
                    result_support = candidates_support
            end = time.time()
            te += end - start

            k += 1
        print('t1')
        print(t1)
        print('t2')
        print(t2)
        print('t3')
        print(t3)
        print('cand')
        print(tg)
        print('support')
        print(ts)
        print('reductions')
        print(tr)
        print('t4')
        print(t4)
        print('t5')
        print(t5)
        print('te')
        print(te)
        return result, result_support

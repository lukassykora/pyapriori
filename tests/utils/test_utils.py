import unittest

import cupy as cp
import numpy as np
from cupyx.scipy.sparse import csr_matrix as cupy_csr_matrix
from parameterized import parameterized
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix

from pyapriori import utils


class TestUtils(unittest.TestCase):
    def test_frequent_single_itemsets(self, type_array):
        transactions = [
            [True, True, True, False, False, False],
            [True, True, True, False, False, False],
            [True, True, True, False, False, False],
            [True, True, False, True, False, False],
            [True, False, False, False, True, True],
            [True, True, True, False, False, False],
            [True, True, True, False, False, False],
        ]
        data_transactions = type_array(transactions)
        itemsets, support, data = utils.frequent_single_itemsets(data_transactions, 2)

        self.assertEqual((7, 3), data.shape)
        self.assertEqual(type(data_transactions), type(data))
        self.assertEqual(3, len(itemsets))
        self.assertEqual(3, len(support))
        expected_itemset = [
            [False, False, True, False, False, False],
            [False, True, False, False, False, False],
            [True, False, False, False, False, False],
        ]
        self.assertEqual(expected_itemset, itemsets.tolist())
        self.assertEqual([5, 6, 7], support.tolist())

    @parameterized.expand(
        [
            (np.array([{2}, {1}, {0}]), None, [0, 0, 1], [1, 2, 2]),
            (np.array([{2, 1}, {2, 0}, {1, 0}]), [0, 0, 1], [0], [1]),
        ]
    )
    def test_generate_candidates(
        self,
        previous_candidates,
        previous_multiplier_mask,
        expected_multiplier_mask_left,
        expected_multiplier_mask_right,
    ):
        multiplier_mask_left, multiplier_mask_right = utils.generate_candidates(
            previous_candidates, previous_multiplier_mask
        )

        np.testing.assert_array_equal(
            expected_multiplier_mask_left, multiplier_mask_left
        )
        np.testing.assert_array_equal(
            expected_multiplier_mask_right, multiplier_mask_right
        )

    @parameterized.expand(
        [
            (
                # data
                np.array(
                    [
                        [True, False, True],
                        [False, True, True],
                        [False, True, True],
                    ]
                ),
                # multiplier_mask_left
                [0, 0, 1],
                # multiplier_mask_right
                [1, 2, 2],
                # expected_new_data
                np.array(
                    [
                        [False, True, False],
                        [False, False, True],
                        [False, False, True],
                    ]
                ),
                # expected_new_data_support
                np.array([0, 1, 2]),
            ),
            (
                # data
                np.array(
                    [
                        [True, False, True],
                        [False, True, True],
                        [False, True, True],
                    ]
                ),
                # multiplier_mask_left
                [0],
                # multiplier_mask_right
                [1],
                # expected_new_data
                np.array(
                    [
                        [False],
                        [False],
                        [False],
                    ]
                ),
                # expected_new_data_support
                np.array([0]),
            ),
        ]
    )
    def test_itemsets_support(
        self,
        data,
        multiplier_mask_left,
        multiplier_mask_right,
        expected_new_data,
        expected_new_data_support,
    ):
        new_data, new_data_support = utils.itemsets_support(
            data, multiplier_mask_left, multiplier_mask_right
        )

        np.testing.assert_array_equal(expected_new_data, new_data)
        np.testing.assert_array_equal(expected_new_data_support, new_data_support)

    @parameterized.expand(
        [
            (
                # candidates
                np.array(
                    [
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ),
                # candidates_support
                np.array([3, 0, 2]),
                # data
                np.array(
                    [
                        [True, False, True],
                        [True, False, True],
                        [True, False, False],
                    ]
                ),
                # multiplier_mask_left
                [0, 0, 1],
                # multiplier_mask_right
                [1, 2, 2],
                # min_support
                2,
                # expected_new_candidates
                np.array(
                    [[0.0, 1.0, 1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]
                ),
                # expected_new_candidates_support
                np.array([3, 2]),
                # expected_new_multiplier_mask_left
                [0, 1],
                # expected_new_data
                np.array(
                    [
                        [True, True],
                        [True, True],
                        [True, False],
                    ]
                ),
            ),
            (
                # candidates
                np.array(
                    [
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ),
                # candidates_support
                np.array([0, 0, 1]),
                # data
                np.array(
                    [
                        [False, False, True],
                        [False, False, False],
                        [False, False, False],
                    ]
                ),
                # multiplier_mask_left
                [0, 0, 1],
                # multiplier_mask_right
                [1, 2, 2],
                # min_support
                2,
                # expected_new_candidates
                np.array([]),
                # expected_new_candidates_support
                np.array([]),
                # expected_new_multiplier_mask_left
                [],
                # expected_new_data
                np.array([[], [], []]),
            ),
        ]
    )
    def test_min_support_set(
        self,
        candidates,
        candidates_support,
        data,
        multiplier_mask_left,
        multiplier_mask_right,
        min_support,
        expected_new_candidates,
        expected_new_candidates_support,
        expected_new_multiplier_mask_left,
        expected_new_data,
    ):
        (
            new_candidates,
            new_candidates_support,
            new_multiplier_mask_left,
            new_data,
        ) = utils.min_support_set(
            candidates,
            candidates_support,
            data,
            multiplier_mask_left,
            multiplier_mask_right,
            min_support,
        )

        self.assertEqual(expected_new_candidates.tolist(), new_candidates.tolist())
        np.testing.assert_array_equal(
            expected_new_candidates_support, new_candidates_support
        )
        self.assertEqual(expected_new_multiplier_mask_left, new_multiplier_mask_left)
        np.testing.assert_array_equal(expected_new_data, new_data)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python

"""Tests for `pyapriori` package."""

import unittest
from pyapriori import PyApriori
import numpy as np


class TestPyApriori(unittest.TestCase):
    def test_pyapriori_numpy_array(self):
        transactions = np.array([
            [True, True, True, False, False, False],
            [True, True, True, False, False, False],
            [True, True, True, False, False, False],
            [True, True, False, True, False, False],
            [True, False, False, False, True, True],
            [True, True, True, False, False, False],
            [True, True, True, False, False, False],
        ])
        py_apriori = PyApriori(2, 2)
        itemsets, support = py_apriori.fit(transactions)

        expected_itemsets = np.array(
            [
                {1, 2},
                {0, 2},
                {0, 1},
                {0, 1, 2}
            ]
        )
        self.assertTrue(np.array_equal(expected_itemsets, itemsets))

        expected_support = np.array([5, 5, 6, 5])
        self.assertTrue(np.array_equal(expected_support, support))


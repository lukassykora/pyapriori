#!/usr/bin/env python

"""Tests for `pyapriori` package."""

import unittest
import numpy as np
from pyapriori import PyApriori


class TestPyApriori(unittest.TestCase):
    def test_pyapriori(self):
        transactions = [
            {0, 1, 2},
            {0, 1, 2},
            {0, 1, 2},
            {0, 1, 3},
            {0, 4, 5},
            {0, 1, 2},
            {0, 1, 2},
        ]
        data_transactions = np.array(transactions)
        py_apriori = PyApriori(2, 2)
        itemsets, support = py_apriori.fit(data_transactions)

        expected_itemsets = np.array(
            [
                {1, 2},
                {0, 2},
                {0, 1},
                {0, 1, 2},
            ]
        )
        np.testing.assert_array_equal(expected_itemsets, itemsets)

        expected_support = np.array([5, 5, 6, 5])
        np.testing.assert_array_equal(expected_support, support)

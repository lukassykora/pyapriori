#!/usr/bin/env python

"""Tests for `pyapriori` package."""

import unittest
from parameterized import parameterized
import numpy as np
import cupy as cp
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from cupyx.scipy.sparse import csr_matrix as cupy_csr_matrix
from pyapriori import PyApriori
from pyapriori.utils.utils import get_numpy_or_cupy


class TestPyApriori(unittest.TestCase):
    @parameterized.expand([
        (np.array,),
        (cp.array,),
        (csr_matrix,),
        (csc_matrix,),
        (lambda x: cupy_csr_matrix(csr_matrix(x)),),
    ])
    def test_pyapriori(self, type_array):
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
        numpy_or_cupy = get_numpy_or_cupy(data_transactions)
        py_apriori = PyApriori(2, 2)
        itemsets, support = py_apriori.fit(data_transactions)

        expected_itemsets = np.array(
            [
                {1, 2},
                {0, 2},
                {0, 1},
                {0, 1, 2}
            ]
        )
        self.assertTrue(np.array_equal(expected_itemsets, itemsets))

        expected_support = numpy_or_cupy.array([5, 5, 6, 5])
        self.assertTrue(numpy_or_cupy.array_equal(expected_support, support))


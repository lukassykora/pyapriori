#!/usr/bin/env python

"""Tests for `pyapriori` package."""

import unittest

import cupy as cp
import numpy as np
from cupyx.scipy.sparse import csr_matrix as cupy_csr_matrix
from parameterized import parameterized
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix

from pyapriori import PyApriori
from pyapriori.utils.utils import get_numpy_or_cupy


class TestPyApriori(unittest.TestCase):
    @parameterized.expand(
        [
            (np.array,),
            (cp.array,),
            (csr_matrix,),
            (csc_matrix,),
            (lambda x: cupy_csr_matrix(csr_matrix(x)),),
        ]
    )
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

        expected_itemsets = numpy_or_cupy.array(
            [
                [False, True, True, False, False, False],
                [True, False, True, False, False, False],
                [True, True, False, False, False, False],
                [True, True, True, False, False, False],
            ]
        )
        numpy_or_cupy.testing.assert_array_equal(expected_itemsets, itemsets)

        expected_support = numpy_or_cupy.array([5, 5, 6, 5])
        numpy_or_cupy.testing.assert_array_equal(expected_support, support)

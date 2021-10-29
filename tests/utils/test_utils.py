import unittest
from parameterized import parameterized
import numpy as np
import cupy as cp
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from cupyx.scipy.sparse import csr_matrix as cupy_csr_matrix
from pyapriori import utils


class TestUtils(unittest.TestCase):
    @parameterized.expand([
        (np.array,),
        (cp.array,),
        (csr_matrix,),
        (csc_matrix,),
        (lambda x: cupy_csr_matrix(csr_matrix(x)),),
    ])
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

        self.assertEqual((7,3), data.shape)
        self.assertEqual(type(data_transactions), type(data))
        self.assertEqual(3, len(itemsets))
        self.assertEqual(3, len(support))
        self.assertEqual([2, 1, 0], itemsets.tolist())
        self.assertEqual([5, 6, 7], support.tolist())


if __name__ == '__main__':
    unittest.main()

import unittest
from StockDataset import get_split_idx_and_length, sliding_windows
import numpy as np


class TestStockDataset(unittest.TestCase):
    def test_get_split_idx_and_length(self):
        data = np.zeros(10)
        idx, length = get_split_idx_and_length(
            is_train=True, data=data, train_split=0.8
        )
        self.assertEqual(idx, 0)
        self.assertEqual(length, 8)
        idx, length = get_split_idx_and_length(
            is_train=False, data=data, train_split=0.8
        )
        self.assertEqual(idx, 8)
        self.assertEqual(length, 2)

        data = np.zeros(11)
        idx, length = get_split_idx_and_length(
            is_train=True, data=data, train_split=0.8
        )
        self.assertEqual(idx, 0)
        self.assertEqual(length, 9)
        idx, length = get_split_idx_and_length(
            is_train=False, data=data, train_split=0.8
        )
        self.assertEqual(idx, 9)
        self.assertEqual(length, 2)

        data = np.zeros(13)
        idx, length = get_split_idx_and_length(
            is_train=True, data=data, train_split=0.8
        )
        self.assertEqual(idx, 0)
        self.assertEqual(length, 10)
        idx, length = get_split_idx_and_length(
            is_train=False, data=data, train_split=0.8
        )
        self.assertEqual(idx, 10)
        self.assertEqual(length, 3)

        data = np.zeros(6)
        idx, length = get_split_idx_and_length(
            is_train=True, data=data, train_split=0.75
        )
        self.assertEqual(idx, 0)
        self.assertEqual(length, 4)
        idx, length = get_split_idx_and_length(
            is_train=False, data=data, train_split=0.75
        )
        self.assertEqual(idx, 4)
        self.assertEqual(length, 2)

    def test_sliding_windows(self):
        data = np.array([0, 1, 2])

        result = sliding_windows(data=data, lookback=2)
        expected = np.array([[0, 1], [1, 2]])

        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
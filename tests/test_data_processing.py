import unittest
from quantum_hybrid_model.data_processing import generate_data


class TestDataProcessing(unittest.TestCase):
    def test_generate_data(self):
        X_train, X_test, y_train, y_test = generate_data()
        self.assertEqual(X_train.shape[1], 4)
        self.assertEqual(X_test.shape[1], 4)
        self.assertEqual(X_train.shape[0], 80)
        self.assertEqual(X_test.shape[0], 20)
        self.assertEqual(len(y_train), 80)
        self.assertEqual(len(y_test), 20)


if __name__ == "__main__":
    unittest.main()

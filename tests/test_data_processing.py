import unittest
from quantum_hybrid_model.data_processing import generate_data, load_real_world_data
import pandas as pd


class TestDataProcessing(unittest.TestCase):
    def test_generate_data(self):
        X_train, X_test, y_train, y_test = generate_data()
        self.assertEqual(X_train.shape[1], 4)
        self.assertEqual(X_test.shape[1], 4)
        self.assertEqual(X_train.shape[0], 80)
        self.assertEqual(X_test.shape[0], 20)
        self.assertEqual(len(y_train), 80)
        self.assertEqual(len(y_test), 20)

    def test_load_real_world_data(self, tmp_path):
        # Create dummy CSV
        df = pd.DataFrame({
            'f1': [0.1, 0.2, 0.3, 0.4],
            'f2': [1, 2, 3, 4],
            'label': [0, 1, 0, 1]
        })
        csv_path = tmp_path / "dummy.csv"
        df.to_csv(csv_path, index=False)
        X_train, X_test, y_train, y_test = load_real_world_data(str(csv_path), test_size=0.5)
        self.assertEqual(X_train.shape[1], 2)
        self.assertEqual(y_train.shape[1], 1)


if __name__ == "__main__":
    unittest.main()

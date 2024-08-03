import unittest
import os
from quantum_hybrid_model.model import initialize_model
from quantum_hybrid_model.data_processing import generate_data
from quantum_hybrid_model.training import train_model, evaluate_model


class TestTraining(unittest.TestCase):
    def setUp(self):
        self.n_qubits = 4
        self.weight_shapes = {"weights": (20, self.n_qubits, 3)}
        self.model = initialize_model(self.n_qubits, self.weight_shapes)
        self.X_train, self.X_test, self.y_train, self.y_test = generate_data()
        self.plot_dir = "./plots/test"
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

    def tearDown(self):
        # Clean up the plot directory after tests
        if os.path.exists(self.plot_dir):
            for file in os.listdir(self.plot_dir):
                file_path = os.path.join(self.plot_dir, file)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            os.rmdir(self.plot_dir)

    def test_train_model(self):
        losses, accuracies = train_model(
            self.model, self.X_train, self.y_train, n_epochs=10, plot_dir=self.plot_dir
        )
        self.assertEqual(len(losses), 10)
        self.assertEqual(len(accuracies), 10)

    def test_evaluate_model(self):
        accuracy = evaluate_model(
            self.model, self.X_test, self.y_test, plot_dir=self.plot_dir
        )
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)


if __name__ == "__main__":
    unittest.main()

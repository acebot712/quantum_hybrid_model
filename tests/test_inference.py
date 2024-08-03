import unittest
import torch
import os
from quantum_hybrid_model.inference import load_model, inference
from quantum_hybrid_model.data_processing import generate_data
from quantum_hybrid_model.model import initialize_model

class TestInference(unittest.TestCase):
    def setUp(self):
        self.model_path = "model.pth"
        self.n_qubits = 4
        # Create a dummy model file
        if not os.path.exists(self.model_path):
            model = initialize_model(self.n_qubits, {"weights": (20, self.n_qubits, 3)})
            torch.save(model.state_dict(), self.model_path)

    def tearDown(self):
        # Remove the dummy model file after tests
        if os.path.exists(self.model_path):
            os.remove(self.model_path)

    def test_load_model(self):
        model = load_model(self.model_path, self.n_qubits)
        self.assertIsNotNone(model)

    def test_inference(self):
        model = load_model(self.model_path, self.n_qubits)
        _, X_test, _, _ = generate_data()
        predictions = inference(model, X_test)
        self.assertEqual(predictions.shape[0], X_test.shape[0])

if __name__ == "__main__":
    unittest.main()

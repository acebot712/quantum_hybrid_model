import unittest
from quantum_hybrid_model.model import initialize_model

def test_plugin_circuit():
    model = initialize_model(n_qubits=4, weight_shapes={"weights": (20, 4, 3)}, quantum_circuit_name="example_custom_circuit")
    assert hasattr(model.quantum, 'qnode')

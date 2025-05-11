import torch
import torch.nn as nn
import pennylane as qml
from .quantum_circuit import create_qnode, advanced_quantum_circuit
import importlib


class AdvancedQuantumLayer(nn.Module):
    def __init__(self, qnode, weight_shapes):
        super(AdvancedQuantumLayer, self).__init__()
        self.qnode = qnode
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

    def forward(self, x):
        return self.qlayer(x)


def get_quantum_circuit_by_name(name):
    if name == "advanced_quantum_circuit":
        return advanced_quantum_circuit
    # Try to load from plugins
    try:
        plugin_module = importlib.import_module(f"plugins.{name}")
        return getattr(plugin_module, name)
    except (ModuleNotFoundError, AttributeError):
        raise ValueError(f"Quantum circuit '{name}' not found.")


class AdvancedHybridModel(nn.Module):
    def __init__(self, n_qubits, weight_shapes, quantum_circuit_name="advanced_quantum_circuit"):
        super(AdvancedHybridModel, self).__init__()
        self.fc1 = nn.Linear(4, n_qubits)
        quantum_circuit = get_quantum_circuit_by_name(quantum_circuit_name)
        self.quantum = AdvancedQuantumLayer(create_qnode(n_qubits, weight_shapes, quantum_circuit), weight_shapes)
        self.fc2 = nn.Linear(n_qubits, 2)
        self.fc3 = nn.Linear(2, 1)

        # Initialize weights
        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.kaiming_uniform_(self.fc2.weight)
        nn.init.kaiming_uniform_(self.fc3.weight)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.quantum(x)
        x = torch.tanh(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


def initialize_model(n_qubits=4, weight_shapes=None, quantum_circuit_name="advanced_quantum_circuit"):
    if weight_shapes is None:
        # Set default weight shapes
        weight_shapes = {"weights": (20, n_qubits, 3)}
    model = AdvancedHybridModel(n_qubits, weight_shapes, quantum_circuit_name)
    return model

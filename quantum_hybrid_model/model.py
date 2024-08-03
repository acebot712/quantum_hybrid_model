import torch
import torch.nn as nn
import pennylane as qml
from .quantum_circuit import create_qnode


class AdvancedQuantumLayer(nn.Module):
    def __init__(self, qnode, weight_shapes):
        super(AdvancedQuantumLayer, self).__init__()
        self.qnode = qnode
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

    def forward(self, x):
        return self.qlayer(x)


class AdvancedHybridModel(nn.Module):
    def __init__(self, n_qubits, weight_shapes):
        super(AdvancedHybridModel, self).__init__()
        self.fc1 = nn.Linear(
            4, n_qubits
        )  # Ensure the input dimension matches the number of qubits
        self.quantum = AdvancedQuantumLayer(
            create_qnode(n_qubits, weight_shapes), weight_shapes
        )
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


def initialize_model(n_qubits=4, weight_shapes=None):
    if weight_shapes is None:
        weight_shapes = {"weights": (20, n_qubits, 3)}  # Set default weight shapes
    model = AdvancedHybridModel(n_qubits, weight_shapes)
    return model

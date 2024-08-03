import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Define the advanced quantum circuit
def advanced_quantum_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]


# Create a QNode
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)
weight_shapes = {"weights": (20, n_qubits, 3)}  # 20 layers, 4 qubits, 3 parameters each
print(f"Weight shapes: {weight_shapes}")


@qml.qnode(dev, interface="torch")
def qnode(inputs, weights):
    return advanced_quantum_circuit(inputs, weights)


# Define the quantum layer
class AdvancedQuantumLayer(nn.Module):
    def __init__(self, qnode, weight_shapes):
        super(AdvancedQuantumLayer, self).__init__()
        self.qnode = qnode
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

    def forward(self, x):
        return self.qlayer(x)


# Define the hybrid model
class AdvancedHybridModel(nn.Module):
    def __init__(self):
        super(AdvancedHybridModel, self).__init__()
        self.fc1 = nn.Linear(
            4, n_qubits
        )  # Ensure the input dimension matches the number of qubits
        self.quantum = AdvancedQuantumLayer(qnode, weight_shapes)
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


# Instantiate the model
model = AdvancedHybridModel()

# Generate synthetic data
X, y = make_classification(
    n_samples=100, n_features=4, n_informative=2, n_redundant=0, random_state=42
)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to PyTorch tensors and ensure float32 dtype
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

# Define loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Lower learning rate

# Training loop
n_epochs = 100
for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    predictions = (predictions > 0.5).float()
    accuracy = (predictions == y_test).float().mean()
    print(f"Accuracy: {accuracy:.4f}")

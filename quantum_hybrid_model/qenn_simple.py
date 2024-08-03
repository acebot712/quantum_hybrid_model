import torch
import torch.nn as nn
import pennylane as qml


class QuantumLayer(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super(QuantumLayer, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=range(self.n_qubits))
            qml.templates.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)]

        weight_shapes = {"weights": (self.n_layers, self.n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x):
        x = self.qlayer(x)
        return x


# Example usage:
n_qubits = 4
n_layers = 2
quantum_layer = QuantumLayer(n_qubits, n_layers)


class HybridModel(nn.Module):
    def __init__(self):
        super(HybridModel, self).__init__()
        self.fc1 = nn.Linear(4, 4)  # Adjust input size if needed
        self.quantum = QuantumLayer(4, 2)  # 4 qubits, 2 layers
        self.fc2 = nn.Linear(4, 2)  # Adjust output size if needed
        self.fc3 = nn.Linear(2, 1)  # Final output layer for binary classification

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.quantum(x)
        x = torch.tanh(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


# Example usage:
model = HybridModel()

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate synthetic data
X, y = make_classification(
    n_samples=100, n_features=4, n_informative=2, n_redundant=0, random_state=42
)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


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

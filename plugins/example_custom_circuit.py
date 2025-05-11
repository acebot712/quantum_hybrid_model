import pennylane as qml

def example_custom_circuit(inputs, weights, n_qubits):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    for i in range(n_qubits):
        qml.RY(weights[0][i][0], wires=i)
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)] 
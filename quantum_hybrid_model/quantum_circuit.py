import pennylane as qml


# Define the advanced quantum circuit
def advanced_quantum_circuit(inputs, weights, n_qubits):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]


def create_qnode(n_qubits, weight_shapes, quantum_circuit=advanced_quantum_circuit):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def qnode(inputs, weights):
        return quantum_circuit(inputs, weights, n_qubits)

    return qnode

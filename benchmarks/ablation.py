import yaml
import os
from quantum_hybrid_model.model import initialize_model
from quantum_hybrid_model.data_processing import get_data_from_config
from quantum_hybrid_model.training import train_model, evaluate_model
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def ablation_study(config):
    X_train, X_test, y_train, y_test = get_data_from_config(config)
    results = {}
    # Full hybrid model
    model = initialize_model(
        n_qubits=config['model']['n_qubits'],
        weight_shapes=config['model']['weight_shapes'],
        quantum_circuit_name=config['model'].get('quantum_circuit', 'advanced_quantum_circuit')
    )
    train_model(model, X_train, y_train, n_epochs=10, learning_rate=0.001)
    acc = evaluate_model(model, X_test, y_test)
    results['full_hybrid'] = acc
    # Classical only (no quantum layer)
    clf = MLPClassifier(max_iter=10)
    clf.fit(X_train.numpy(), y_train.numpy().ravel())
    acc_classical = accuracy_score(y_test.numpy(), clf.predict(X_test.numpy()))
    results['classical_only'] = acc_classical
    # Alternative quantum circuit (if available)
    try:
        model_alt = initialize_model(
            n_qubits=config['model']['n_qubits'],
            weight_shapes=config['model']['weight_shapes'],
            quantum_circuit_name='example_custom_circuit'
        )
        train_model(model_alt, X_train, y_train, n_epochs=10, learning_rate=0.001)
        acc_alt = evaluate_model(model_alt, X_test, y_test)
        results['alt_quantum_circuit'] = acc_alt
    except Exception as e:
        results['alt_quantum_circuit'] = str(e)
    print("Ablation results:", results)
    return results

def main():
    config_path = os.environ.get('QENN_CONFIG', 'configs/experiment_template.yaml')
    config = load_config(config_path)
    ablation_study(config)

if __name__ == "__main__":
    main() 
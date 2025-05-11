import yaml
import os
import torch
from quantum_hybrid_model.model import initialize_model
from quantum_hybrid_model.data_processing import generate_data, get_data_from_config
from quantum_hybrid_model.training import train_model, evaluate_model, run_optuna_optimization
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_baseline(X_train, X_test, y_train, y_test, config):
    clf = MLPClassifier(max_iter=config['training']['n_epochs'])
    clf.fit(X_train, y_train.ravel())
    preds = clf.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, preds),
        'roc_auc': roc_auc_score(y_test, preds),
        'f1': f1_score(y_test, preds)
    }
    print("Baseline (MLPClassifier) metrics:", metrics)
    return metrics

def run_quantum_model(config):
    X_train, X_test, y_train, y_test = get_data_from_config(config)
    model = initialize_model(
        n_qubits=config['model']['n_qubits'],
        weight_shapes=config['model']['weight_shapes'],
        quantum_circuit_name=config['model'].get('quantum_circuit', 'advanced_quantum_circuit')
    )
    # Training
    train_model(
        model, X_train, y_train,
        n_epochs=config['training']['n_epochs'],
        learning_rate=config['training']['learning_rate'],
        save_path=config['training']['save_path'],
        plot_dir=config['visualization']['plot_dir']
    )
    # Evaluation
    acc = evaluate_model(model, X_test, y_test, plot_dir=config['visualization']['plot_dir'])
    return acc

def main():
    config_path = os.environ.get('QENN_CONFIG', 'configs/experiment_template.yaml')
    config = load_config(config_path)
    print(f"Loaded config: {config_path}")
    X_train, X_test, y_train, y_test = get_data_from_config(config)
    if config.get('hyperparameter_optimization', {}).get('enabled', False):
        run_optuna_optimization(config, X_train, y_train, X_test, y_test)
    # Baseline
    if config['benchmark']['run_baseline']:
        run_baseline(X_train.numpy(), X_test.numpy(), y_train.numpy(), y_test.numpy(), config)
    # Quantum Model
    run_quantum_model(config)

if __name__ == "__main__":
    main() 
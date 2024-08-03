import torch
from .model import initialize_model

def load_model(model_path, n_qubits=4):
    weight_shapes = {
        "weights": (20, n_qubits, 3)
    }  # Define weight_shapes inside the function
    model = initialize_model(n_qubits, weight_shapes)
    state_dict = torch.load(model_path, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def inference(model, X):
    with torch.no_grad():
        predictions = model(X)
        predictions = (predictions > 0.5).float()
    return predictions

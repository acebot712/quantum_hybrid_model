import argparse
from .data_processing import generate_data
from .model import initialize_model
from .training import train_model, evaluate_model
from .inference import load_model, inference


def main(mode, args):
    # Define number of qubits and weight shapes
    n_qubits = args.n_qubits
    # 20 layers, 4 qubits, 3 parameters each
    weight_shapes = {"weights": (20, n_qubits, 3)}

    if mode == 'train':
        # Generate data
        X_train, X_test, y_train, y_test = generate_data()

        # Initialize model
        model = initialize_model(n_qubits, weight_shapes)

        # Train model and get metrics
        losses, accuracies = train_model(
            model,
            X_train,
            y_train,
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            save_path=args.save_path,
            plot_dir=args.plot_dir
        )

        # Evaluate model
        test_accuracy = evaluate_model(
            model, X_test, y_test, plot_dir=args.plot_dir)

        # Print the evaluation results
        print(f"Final Test Accuracy: {test_accuracy:.4f}")

    elif mode == 'inference':
        # Load model
        model = load_model(args.model_path, n_qubits)

        # Generate data (or load new data for inference)
        _, X_test, _, y_test = generate_data()

        # Perform inference
        predictions = inference(model, X_test)
        print("Predictions:", predictions)

    else:
        print(f"Unknown mode: {mode}")


def cli():
    parser = argparse.ArgumentParser(
        description="Quantum Enhanced Neural Network (QENN)")
    parser.add_argument(
        "mode",
        choices=[
            "train",
            "inference"],
        help="Mode to run the QENN: train or inference")

    # Training arguments
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=100,
        help="Number of epochs for training")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for training")
    parser.add_argument(
        "--save_path",
        type=str,
        default="model.pth",
        help="Path to save the trained model")

    # Shared arguments
    parser.add_argument(
        "--n_qubits",
        type=int,
        default=4,
        help="Number of qubits in the quantum circuit")
    parser.add_argument(
        "--plot_dir",
        type=str,
        default=".",
        help="Directory to save plots and metrics")

    # Inference arguments
    parser.add_argument(
        "--model_path",
        type=str,
        default="model.pth",
        help="Path to the trained model for inference")

    args = parser.parse_args()
    main(args.mode, args)


if __name__ == "__main__":
    cli()

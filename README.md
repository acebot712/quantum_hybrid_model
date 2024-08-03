# Quantum Enhanced Neural Network (QENN)

This project implements a Quantum Enhanced Neural Network (QENN), combining classical neural networks with quantum circuits to potentially enhance model performance on specific tasks.

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Project Structure

The project consists of the following main components:

- `main.py`: Entry point of the application, handling command-line arguments and orchestrating training and inference processes.
- `data_processing.py`: Responsible for generating or loading the dataset.
- `model.py`: Defines the structure of the quantum-classical hybrid model.
- `training.py`: Contains functions for training and evaluating the model.
- `inference.py`: Provides functionality for loading a trained model and performing inference.
- `plot_utils.py`: Utility functions for creating various plots and visualizations.

## Installation

To run this project, you need Docker installed on your system. The Dockerfile in the project root contains all necessary dependencies.

1. Pull the latest docker image from the GitHub Container Registry:
   ```sh
   docker pull ghcr.io/acebot712/quantum_hybrid_model:latest
   ```

   Alternatively, build the Docker image locally:
   ```sh
   docker build -t quantum_hybrid_model .
   ```

2. To open an interactive terminal into the container with the QENN CLI installed:
   ```sh
   docker run -it --entrypoint /bin/bash quantum_hybrid_model
   ```
   You can run `qenn` CLI commands inside this container.

## Usage

The application can be run in two modes: training and inference.

### Training

To train the model, use the following command:

```sh
docker run -v $(pwd):/app quantum_hybrid_model:latest qenn train \
  --n_epochs 100 \
  --learning_rate 0.0005 \
  --save_path /app/my_model.pth \
  --n_qubits 5 \
  --plot_dir /app/plots/train
```

Arguments:
- `--n_epochs`: Number of training epochs (default: 100)
- `--learning_rate`: Learning rate for the optimizer (default: 0.001)
- `--save_path`: Path to save the trained model (default: /app/model.pth)
- `--n_qubits`: Number of qubits in the quantum circuit (default: 4)
- `--plot_dir`: Directory to save plots and metrics (default: /app)

### Inference

To perform inference using a trained model, use the following command:

```sh
docker run -v $(pwd):/app quantum_hybrid_model:latest qenn inference \
  --model_path /app/my_model.pth \
  --n_qubits 5 \
  --plot_dir /app/plots/inference
```

Arguments:
- `--model_path`: Path to the trained model file (default: /app/model.pth)
- `--n_qubits`: Number of qubits in the quantum circuit (default: 4)
- `--plot_dir`: Directory to save plots and metrics (default: /app)

## Model Architecture

The QENN model consists of a quantum circuit followed by classical neural network layers. The quantum circuit uses a specified number of qubits and applies a series of parameterized quantum operations. The output of the quantum circuit is then processed by classical layers to produce the final prediction.

## Training Process

The training process includes the following steps:

1. Data generation or loading
2. Model initialization
3. Training loop with periodic logging of loss and accuracy
4. Model evaluation on the test set
5. Saving of the trained model
6. Generation of training plots (loss and accuracy curves)

## Evaluation Metrics

The model is evaluated using the following metrics:

- Accuracy
- ROC AUC score
- Classification report (precision, recall, F1-score)

Additionally, the following visualizations are generated:

- ROC AUC curve
- Confusion matrix
- Training loss and accuracy plots

## Results

After training, the model's performance metrics and visualizations can be found in the specified plot directory. The `metrics.json` file contains detailed classification metrics and the ROC AUC score.

## Future Work

Potential areas for improvement and expansion:

1. Experiment with different quantum circuit architectures
2. Implement data preprocessing techniques
3. Add support for custom datasets
4. Explore hyperparameter tuning
5. Implement additional quantum-classical hybrid architectures

## Contributing

Contributions to this project are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
5. Push to the branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

Please ensure that your code adheres to the existing style and includes appropriate tests and documentation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

If you encounter any issues or have questions, please file an issue on the GitHub repository.
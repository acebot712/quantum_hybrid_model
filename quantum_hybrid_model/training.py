import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, roc_auc_score
from .plot_utils import (
    create_training_plots,
    generate_roc_auc_curve,
    generate_confusion_matrix,
)
import os
import json


def train_model(
    model,
    X_train,
    y_train,
    n_epochs=100,
    learning_rate=0.001,
    save_path="model.pth",
    plot_dir=".",
):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize lists to store metrics
    epoch_losses = []
    epoch_accuracies = []

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Store the loss
        epoch_losses.append(loss.item())

        # Calculate and store the accuracy
        model.eval()
        with torch.no_grad():
            predictions = model(X_train)
            predictions = (predictions > 0.5).float()
            accuracy = (predictions == y_train).float().mean().item()
            epoch_accuracies.append(accuracy)

        if (epoch + 1) % 10 == 0:
            print(
                f"""
                  Epoch {epoch+1}/{n_epochs}, \ 
                  Loss: {loss.item():.4f}, \ 
                  Accuracy: {accuracy:.4f}
                """
            )

    # Save the model
    torch.save(model.state_dict(), save_path)

    # Generate loss and accuracy graphs
    create_training_plots(n_epochs, epoch_losses, epoch_accuracies, plot_dir)

    return epoch_losses, epoch_accuracies


def evaluate_model(model, X_test, y_test, plot_dir="."):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        predicted_labels = (predictions > 0.5).float()
        accuracy = (predicted_labels == y_test).float().mean().item()
        roc_auc = roc_auc_score(y_test, predictions)
        class_report = classification_report(
            y_test, predicted_labels, output_dict=True, zero_division=1
        )
        class_report_str = classification_report(
            y_test, predicted_labels, zero_division=1
        )

        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(class_report_str)

    # Save metrics to a file
    metrics_path = os.path.join(plot_dir, "metrics.json")
    class_report["roc_auc"] = roc_auc
    with open(metrics_path, "w") as f:
        json.dump(class_report, f)

    # Generate ROC AUC curve
    generate_roc_auc_curve(y_test, predictions, plot_dir)

    # Generate confusion matrix
    generate_confusion_matrix(y_test, predicted_labels, plot_dir)

    return accuracy

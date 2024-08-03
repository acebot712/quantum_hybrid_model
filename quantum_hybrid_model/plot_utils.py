import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
)
import os


def create_training_plots(
        n_epochs,
        epoch_losses,
        epoch_accuracies,
        plot_dir="."):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plt.figure()
    plt.plot(range(n_epochs), epoch_losses, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "training_loss.png"))

    plt.figure()
    plt.plot(range(n_epochs), epoch_accuracies, label="Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "training_accuracy.png"))


def generate_roc_auc_curve(y_test, predictions, plot_dir="."):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    fpr, tpr, _ = roc_curve(y_test, predictions)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2,
             label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(plot_dir, "roc_auc_curve.png"))


def generate_confusion_matrix(y_test, predictions, plot_dir="."):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(plot_dir, "confusion_matrix.png"))

import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from preprocessing import load_mnist_data, preprocess_mnist_data
from model import load_trained_cnn, load_trained_mlp
from utils import (
    plot_confusion_matrix_figure,
    plot_misclassifications,
    save_metrics_to_json
)

def evaluate_models():
    """
    Evaluates trained CNN and MLP baseline models on the MNIST test set.
    Generates test accuracy, test loss, confusion matrix, classification report (precision, recall, f1-score),
    and saves output artifacts.
    """
    print("Loading MNIST dataset for evaluation...")
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    _, x_test_proc = preprocess_mnist_data(x_train, x_test)

    # Load trained models
    cnn_model = load_trained_cnn()
    mlp_model = load_trained_mlp()

    metrics_results = {
        "dataset": {
            "num_train": int(len(x_train)),
            "num_test": int(len(x_test)),
            "image_dim": "28x28x1",
            "num_classes": 10
        }
    }

    if cnn_model is not None:
        print("Evaluating CNN model...")
        cnn_loss, cnn_acc = cnn_model.evaluate(x_test_proc, y_test, verbose=0)
        y_pred_prob = cnn_model.predict(x_test_proc, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)

        # Confusion Matrix
        cm = plot_confusion_matrix_figure(y_test, y_pred, "outputs/confusion_matrix.png")

        # Misclassification Samples
        misclassified = plot_misclassifications(x_test_proc, y_test, y_pred, y_pred_prob, "outputs/misclassifications.png")

        # Classification Report
        report_dict = classification_report(y_test, y_pred, digits=4, output_dict=True)

        metrics_results["cnn"] = {
            "test_accuracy": float(cnn_acc),
            "test_loss": float(cnn_loss),
            "confusion_matrix": cm.tolist(),
            "classification_report": report_dict,
            "misclassified_samples_count": int(len(np.where(y_test != y_pred)[0])),
            "sample_misclassifications": misclassified
        }
        
        # Save simple metrics text file
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/metrics.txt", "w") as f:
            f.write(f"{cnn_acc}\n{cnn_loss}\n")

        print(f"CNN Test Accuracy: {cnn_acc*100:.2f}% | Test Loss: {cnn_loss:.4f}")
    else:
        print("Warning: CNN model not found for evaluation.")

    if mlp_model is not None:
        print("Evaluating MLP baseline model...")
        mlp_loss, mlp_acc = mlp_model.evaluate(x_test_proc, y_test, verbose=0)
        y_mlp_prob = mlp_model.predict(x_test_proc, verbose=0)
        y_mlp_pred = np.argmax(y_mlp_prob, axis=1)
        mlp_report = classification_report(y_test, y_mlp_pred, digits=4, output_dict=True)

        metrics_results["mlp"] = {
            "test_accuracy": float(mlp_acc),
            "test_loss": float(mlp_loss),
            "classification_report": mlp_report
        }
        print(f"MLP Test Accuracy: {mlp_acc*100:.2f}% | Test Loss: {mlp_loss:.4f}")

    # Save metrics JSON
    save_metrics_to_json(metrics_results, "outputs/metrics.json")
    print("Evaluation results saved to outputs/metrics.json")
    return metrics_results

if __name__ == "__main__":
    evaluate_models()

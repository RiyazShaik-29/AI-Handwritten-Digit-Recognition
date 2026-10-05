import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

def save_metrics_to_json(metrics_dict, output_path="outputs/metrics.json"):
    """
    Saves model evaluation metrics to JSON file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)

def load_metrics_from_json(input_path="outputs/metrics.json"):
    """
    Loads model evaluation metrics from JSON file.
    """
    if os.path.exists(input_path):
        with open(input_path, "r") as f:
            return json.load(f)
    return None

def save_training_history_json(history_dict, output_path="outputs/training_history.json"):
    """
    Saves Keras history dictionary to JSON file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    serializable_history = {}
    for key, val in history_dict.items():
        serializable_history[key] = [float(x) for x in val]
    with open(output_path, "w") as f:
        json.dump(serializable_history, f, indent=4)

def load_training_history_json(input_path="outputs/training_history.json"):
    """
    Loads training history dictionary from JSON file.
    """
    if os.path.exists(input_path):
        with open(input_path, "r") as f:
            return json.load(f)
    return None

def plot_training_curves(history_data, output_dir="outputs"):
    """
    Generates and saves accuracy and loss curves from Keras history object or dictionary.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if hasattr(history_data, 'history'):
        hist = history_data.history
    else:
        hist = history_data

    epochs = range(1, len(hist['accuracy']) + 1)

    # Accuracy Plot
    fig_acc, ax_acc = plt.subplots(figsize=(8, 5))
    ax_acc.plot(epochs, hist['accuracy'], 'o-', label='Training Accuracy', color='#3B82F6', linewidth=2.5, markersize=5)
    ax_acc.plot(epochs, hist['val_accuracy'], 's-', label='Validation Accuracy', color='#10B981', linewidth=2.5, markersize=5)
    ax_acc.set_title('CNN Model Training & Validation Accuracy', fontsize=14, fontweight='bold', pad=15)
    ax_acc.set_xlabel('Epoch', fontsize=11)
    ax_acc.set_ylabel('Accuracy', fontsize=11)
    ax_acc.legend(loc='lower right', frameon=True)
    ax_acc.grid(True, linestyle='--', alpha=0.5)
    fig_acc.tight_layout()
    fig_acc.savefig(os.path.join(output_dir, 'accuracy_curve.png'), dpi=200)
    plt.close(fig_acc)

    # Loss Plot
    fig_loss, ax_loss = plt.subplots(figsize=(8, 5))
    ax_loss.plot(epochs, hist['loss'], 'o-', label='Training Loss', color='#EF4444', linewidth=2.5, markersize=5)
    ax_loss.plot(epochs, hist['val_loss'], 's-', label='Validation Loss', color='#F59E0B', linewidth=2.5, markersize=5)
    ax_loss.set_title('CNN Model Training & Validation Loss', fontsize=14, fontweight='bold', pad=15)
    ax_loss.set_xlabel('Epoch', fontsize=11)
    ax_loss.set_ylabel('Loss', fontsize=11)
    ax_loss.legend(loc='upper right', frameon=True)
    ax_loss.grid(True, linestyle='--', alpha=0.5)
    fig_loss.tight_layout()
    fig_loss.savefig(os.path.join(output_dir, 'loss_curve.png'), dpi=200)
    plt.close(fig_loss)

def plot_confusion_matrix_figure(y_true, y_pred, output_path="outputs/confusion_matrix.png"):
    """
    Generates and saves a 10x10 confusion matrix heatmap.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=[str(i) for i in range(10)],
        yticklabels=[str(i) for i in range(10)],
        cbar_kws={'label': 'Sample Count'},
        ax=ax
    )
    ax.set_title('MNIST CNN Model — Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted Digit Class', fontsize=11, labelpad=10)
    ax.set_ylabel('Actual (True) Digit Class', fontsize=11, labelpad=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return cm

def plot_misclassifications(x_test, y_true, y_pred, y_prob, output_path="outputs/misclassifications.png", max_samples=10):
    """
    Finds and saves visual examples of misclassified test digits.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mis_idx = np.where(y_true != y_pred)[0]
    
    if len(mis_idx) == 0:
        return []
    
    selected = mis_idx[:max_samples]
    cols = 5
    rows = int(np.ceil(len(selected) / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3 * rows))
    fig.suptitle('Sample Misclassified Digits (True vs Predicted)', fontsize=14, fontweight='bold', y=0.98)
    
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    misclassified_data = []
    for i, idx in enumerate(selected):
        img = x_test[idx].squeeze()
        true_lbl = y_true[idx]
        pred_lbl = y_pred[idx]
        conf = y_prob[idx][pred_lbl] * 100
        
        misclassified_data.append({
            "index": int(idx),
            "true_label": int(true_lbl),
            "predicted_label": int(pred_lbl),
            "confidence": float(conf)
        })
        
        axes_flat[i].imshow(img, cmap='gray')
        axes_flat[i].set_title(f"True: {true_lbl} | Pred: {pred_lbl}\n({conf:.1f}% conf)", color='#DC2626', fontsize=10, fontweight='bold')
        axes_flat[i].axis('off')
        
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')
        
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    
    return misclassified_data

def create_interactive_class_distribution_chart(y_train, y_test):
    """
    Creates an interactive Plotly bar chart showing sample counts per class.
    """
    train_counts = np.bincount(y_train, minlength=10)
    test_counts = np.bincount(y_test, minlength=10)
    
    digits = [f"Digit {i}" for i in range(10)]
    
    fig = go.Figure(data=[
        go.Bar(name='Training Set (60,000)', x=digits, y=train_counts, marker_color='#3B82F6'),
        go.Bar(name='Test Set (10,000)', x=digits, y=test_counts, marker_color='#10B981')
    ])
    
    fig.update_layout(
        barmode='group',
        title=dict(text='MNIST Class Distribution (Train vs Test)', font=dict(size=16, color='#1E293B')),
        xaxis=dict(title='Digit Class (0-9)', tickmode='linear'),
        yaxis=dict(title='Number of Samples'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(241,245,249,0.5)',
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def create_interactive_probability_chart(probabilities):
    """
    Creates a sleek horizontal bar chart of 10 class probabilities.
    """
    digits = [f"Digit {i}" for i in range(10)]
    probs = probabilities * 100
    pred_idx = np.argmax(probabilities)
    
    colors = ['#94A3B8'] * 10
    colors[pred_idx] = '#10B981'  # Highlight predicted class in emerald green
    
    fig = go.Figure(go.Bar(
        x=probs,
        y=digits,
        orientation='h',
        marker=dict(color=colors, line=dict(color='#334155', width=1)),
        text=[f"{p:.2f}%" for p in probs],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=dict(text='10-Class Softmax Probability Distribution', font=dict(size=15, color='#1E293B')),
        xaxis=dict(title='Confidence / Probability (%)', range=[0, 115]),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(241,245,249,0.5)',
        height=340,
        margin=dict(l=70, r=40, t=40, b=40)
    )
    return fig

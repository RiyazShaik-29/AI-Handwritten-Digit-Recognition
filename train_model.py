import os
import numpy as np
import tensorflow as tf
from preprocessing import load_mnist_data, preprocess_mnist_data
from model import build_cnn_model, build_mlp_model, compile_model
from utils import plot_training_curves, save_training_history_json
from evaluate_model import evaluate_models

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def train():
    print("=" * 65)
    print("AI-Based Handwritten Digit Recognition System — Training Pipeline")
    print("=" * 65)

    # 1. Load MNIST Dataset
    print("\n[Step 1/6] Loading official MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    print(f"Dataset loaded: {x_train.shape[0]} training samples, {x_test.shape[0]} test samples.")

    # 2. Preprocess Data
    print("\n[Step 2/6] Normalizing pixels (0-255 -> 0.0-1.0) & reshaping to 28x28x1...")
    x_train_proc, x_test_proc = preprocess_mnist_data(x_train, x_test)

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # 3. Build & Compile CNN and MLP models
    print("\n[Step 3/6] Building CNN architecture & MLP baseline...")
    cnn_model = build_cnn_model()
    cnn_model = compile_model(cnn_model)
    cnn_model.summary()

    mlp_model = build_mlp_model()
    mlp_model = compile_model(mlp_model)

    # 4. Train Models
    batch_size = 128
    epochs = 10
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath="models/mnist_cnn.keras",
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    print(f"\n[Step 4/6] Training CNN model for {epochs} epochs (batch_size={batch_size}, val_split=0.1)...")
    cnn_history = cnn_model.fit(
        x_train_proc, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )

    print(f"\nTraining MLP baseline model ({epochs} epochs)...")
    mlp_history = mlp_model.fit(
        x_train_proc, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1
    )

    # 5. Save Models & Artifacts
    print("\n[Step 5/6] Saving trained models and performance plots...")
    cnn_path = "models/mnist_cnn.keras"
    cnn_model.save(cnn_path)
    # Save alias for legacy naming
    cnn_model.save("models/handwritten_digit_cnn.keras")
    print(f"CNN model saved to: {cnn_path}")

    mlp_path = "models/mnist_mlp.keras"
    mlp_model.save(mlp_path)
    mlp_model.save("models/handwritten_digit_mlp.keras")
    print(f"MLP model saved to: {mlp_path}")

    save_training_history_json(cnn_history.history, "outputs/training_history.json")
    plot_training_curves(cnn_history, "outputs")
    print("Saved training loss & accuracy curves to outputs/")

    # 6. Evaluation
    print("\n[Step 6/6] Running evaluation pipeline on test dataset...")
    evaluate_models()

    print("\n" + "=" * 65)
    print("Training Pipeline Complete!")
    print("Start Streamlit App with: python -m streamlit run app.py")
    print("=" * 65)

if __name__ == "__main__":
    train()

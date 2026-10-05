import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def build_cnn_model():
    """
    Builds the primary Convolutional Neural Network (CNN) for MNIST digit recognition.
    Architecture:
    - Input: 28 x 28 x 1
    - Conv2D (32 filters, 3x3, ReLU)
    - MaxPooling2D (2x2)
    - Conv2D (64 filters, 3x3, ReLU)
    - MaxPooling2D (2x2)
    - Flatten
    - Dense (128 units, ReLU)
    - Dropout (0.5)
    - Dense (10 units, Softmax)
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), name="conv2d_1"),
        MaxPooling2D((2, 2), name="maxpool_1"),
        Conv2D(64, (3, 3), activation='relu', name="conv2d_2"),
        MaxPooling2D((2, 2), name="maxpool_2"),
        Flatten(name="flatten"),
        Dense(128, activation='relu', name="dense_128"),
        Dropout(0.5, name="dropout_0.5"),
        Dense(10, activation='softmax', name="output_softmax")
    ], name="MNIST_CNN_Classifier")
    return model

def build_mlp_model():
    """
    Builds a simple Multi-Layer Perceptron (MLP) baseline model for academic comparison.
    Architecture:
    - Input: 28 x 28 x 1
    - Flatten
    - Dense (128 units, ReLU)
    - Dense (64 units, ReLU)
    - Dense (10 units, Softmax)
    """
    model = Sequential([
        Flatten(input_shape=(28, 28, 1), name="mlp_flatten"),
        Dense(128, activation='relu', name="mlp_dense_128"),
        Dense(64, activation='relu', name="mlp_dense_64"),
        Dense(10, activation='softmax', name="mlp_output_softmax")
    ], name="MNIST_MLP_Baseline")
    return model

def compile_model(model, learning_rate=0.001):
    """
    Compiles Keras model with Adam optimizer and Sparse Categorical Crossentropy loss.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def load_trained_cnn(model_path="models/mnist_cnn.keras"):
    """
    Loads saved CNN model if exists.
    Checks primary path 'models/mnist_cnn.keras' and alternative path 'models/handwritten_digit_cnn.keras'.
    """
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    elif os.path.exists("models/handwritten_digit_cnn.keras"):
        return tf.keras.models.load_model("models/handwritten_digit_cnn.keras")
    return None

def load_trained_mlp(model_path="models/mnist_mlp.keras"):
    """
    Loads saved MLP baseline model if exists.
    Checks primary path 'models/mnist_mlp.keras' and alternative path 'models/handwritten_digit_mlp.keras'.
    """
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    elif os.path.exists("models/handwritten_digit_mlp.keras"):
        return tf.keras.models.load_model("models/handwritten_digit_mlp.keras")
    return None

def get_model_summary_info(model):
    """
    Extracts summary parameters and layer architecture details for academic UI display.
    """
    if model is None:
        return None
    
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    layers_info = []
    for layer in model.layers:
        layers_info.append({
            "name": layer.name,
            "type": layer.__class__.__name__,
            "output_shape": str(layer.output_shape),
            "params": layer.count_params()
        })
        
    return {
        "model_name": model.name,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "num_layers": len(model.layers),
        "layers": layers_info
    }

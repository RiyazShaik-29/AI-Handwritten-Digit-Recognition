# AI-Based Handwritten Digit Recognition System
## Academic Major Project Technical Report & Viva Study Notes

### Executive Summary
This project implements a web-based computer vision and deep learning system for handwritten digit recognition using Convolutional Neural Networks (CNN) trained on the benchmark MNIST dataset. The system features interactive drawing canvas input, custom image uploading, automated image preprocessing, confidence scoring, class probability visualizers, dynamic dataset exploration, and an academic viva defense guide.

---

### 1. System Architecture & Lifecycle

```text
[ DATASET ]  --->  [ PREPROCESSING ]  --->  [ CNN MODEL ]  --->  [ EVALUATION ]
  MNIST 70k         Normalize [0-1]          2× Conv2D            Acc & Loss
  28x28 Grayscale   Reshape (28,28,1)        MaxPool, Dropout     Confusion Matrix
                                             Softmax 10          Classification Report
                                                 |
                                                 v
[ PREDICTION ]  <---  [ CV PIPELINE ]  <---  [ USER INPUT ]
  Predicted Digit      Grayscale, Invert      Drawing Canvas /
  Confidence %         Crop & Center 28x28    Image Upload
```

---

### 2. Model Architecture Details

#### Convolutional Neural Network (CNN)
- **Input Layer:** `(28, 28, 1)` grayscale image tensor.
- **Convolutional Layer 1:** 32 filters, 3×3 kernel size, ReLU activation. Captures low-level spatial features (edges, curves).
- **MaxPooling Layer 1:** 2×2 spatial reduction (downsamples spatial dimensions by 50%). Provides translation invariance.
- **Convolutional Layer 2:** 64 filters, 3×3 kernel size, ReLU activation. Captures complex digit structural shapes.
- **MaxPooling Layer 2:** 2×2 spatial reduction.
- **Flatten Layer:** Vectorizes 2D feature maps into a 1D tensor.
- **Dense Layer:** 128 fully connected neurons with ReLU activation.
- **Dropout Layer:** Rate 0.3 (randomly drops 30% of neurons during training to prevent overfitting).
- **Output Layer:** 10 neurons with Softmax activation function.

#### MLP Baseline Comparison
- **Structure:** `Flatten -> Dense(128, ReLU) -> Dropout(0.3) -> Dense(64, ReLU) -> Dense(10, Softmax)`
- **Key Difference:** MLP flattens the 2D image immediately, losing 2D spatial relationships between neighboring pixels. The CNN uses local receptive fields to preserve spatial context.

---

### 3. Preprocessing & Computer Vision Pipeline

User-provided drawings or uploaded images go through a robust 5-step preprocessing pipeline before inference:
1. **Grayscale Conversion:** Converts RGB/RGBA to single-channel 8-bit grayscale (`L`).
2. **Polarity Check & Inversion:** Samples image borders to detect background intensity. If light, inverts pixels so digits are represented as bright strokes (255) on dark backgrounds (0), matching MNIST.
3. **Bounding Box Bounding & Cropping:** Locates non-zero digit pixel bounding box and crops tight bounds around the digit.
4. **Aspect-Ratio Preserving Resize & Center Alignment:** Resizes digit into a 20×20 bounding box while maintaining aspect ratio, then centers it within a 28×28 blank canvas.
5. **Pixel Normalization:** Scales pixel values from `[0, 255]` to `[0.0, 1.0]` float32.

---

### 4. Mathematical Formulation

#### 1. Rectified Linear Unit (ReLU)
$$f(x) = \max(0, x)$$

#### 2. Softmax Activation Function
$$P(y = k \mid \mathbf{x}) = \frac{e^{z_k}}{\sum_{j=0}^{9} e^{z_j}}$$

#### 3. Sparse Categorical Cross-Entropy Loss
$$\mathcal{L} = -\sum_{i=1}^{N} \log P(y_i \mid \mathbf{x}_i)$$

---

### 5. Common Viva Defense Questions & Answers

1. **Why use CNN over traditional ML algorithms like SVM or KNN?**
   CNNs automatically extract hierarchical spatial features directly from raw pixels via learnable convolutional kernels, avoiding manual feature engineering (like HOG or SIFT).

2. **Why normalize pixel values to [0, 1]?**
   Unnormalized inputs (0–255) cause large, unstable gradient updates during backpropagation. Normalization stabilizes training convergence and prevents exploding gradients.

3. **What is the significance of the Confusion Matrix?**
   The confusion matrix reveals specific pairwise misclassifications (e.g., distinguishing between visually similar digits such as 4 and 9, or 3 and 5).

---

### 6. File & Directory Organization

```text
mnist_digit_recognition/
├── app.py                   # Main Streamlit Dashboard Application
├── train_model.py           # Training Pipeline Script (CNN & MLP)
├── evaluate_model.py        # Quantitative Model Evaluation Script
├── preprocessing.py         # Data Loaders & Image Preprocessing Pipeline
├── model.py                 # Neural Network Architecture Definitions
├── utils.py                 # Plotly/Matplotlib Plot Helpers & Metrics Storage
├── models/
│   ├── handwritten_digit_cnn.keras
│   └── handwritten_digit_mlp.keras
├── outputs/
│   ├── accuracy_curve.png
│   ├── loss_curve.png
│   ├── confusion_matrix.png
│   ├── misclassifications.png
│   ├── metrics.json
│   └── metrics.txt
├── requirements.txt         # Project Dependencies
├── README.md                # Project Overview & Setup Instructions
└── report/
    └── project_notes.md     # Technical Documentation & Viva Study Notes
```

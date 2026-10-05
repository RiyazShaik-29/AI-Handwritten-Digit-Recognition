# AI-Based Handwritten Digit Recognition System
## Professional Academic Major Project Implementation

An end-to-end Computer Vision and Deep Learning system for handwritten digit recognition (digits 0–9) powered by a **Convolutional Neural Network (CNN)** trained on the benchmark **MNIST dataset** and served via an interactive **Streamlit** web application.

---

## 📌 Features

1. **Dataset Exploration & Insights:**
   - Interactive breakdown of the 60,000 training and 10,000 test MNIST images.
   - Ground-truth representative sample image visualization.
   - Interactive class distribution charts (Train vs Test set).

2. **Deep Learning Model Architecture:**
   - 2D Convolutional Neural Network (Conv2D -> MaxPool2D -> Conv2D -> MaxPool2D -> Dense -> Dropout -> Softmax).
   - Optional Multi-Layer Perceptron (MLP) baseline model for spatial learning comparison.

3. **Rigorous Evaluation Metrics:**
   - Test Accuracy and Categorical Crossentropy Loss.
   - 10×10 Confusion Matrix Heatmap.
   - Training vs Validation Accuracy and Loss Curves.
   - Per-class Precision, Recall, and F1-Score classification report.

4. **Real-Time Interactive Digit Prediction:**
   - **Option A (Canvas Drawing):** Interactive drawing canvas to draw digits manually with mouse/touch.
   - **Option B (Image Upload):** File uploader supporting PNG, JPG, and JPEG.
   - **Automated Preprocessing Pipeline:** Grayscale conversion, polarity check & inversion (white digit on black background), noise thresholding, bounding box cropping, aspect-ratio preserving scaling to 20×20, and center placement in 28×28 tensor.
   - **Explainable Prediction:** Top predicted digit display, confidence percentage, and full 10-class Softmax probability distribution chart.

---

## 🛠️ Technology Stack

- **Language:** Python 3.10+
- **Deep Learning Framework:** TensorFlow / Keras
- **Web Application Framework:** Streamlit
- **Drawing Canvas:** Streamlit Drawable Canvas
- **Data Manipulation & Processing:** NumPy, Pandas, Pillow
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Machine Learning Evaluation:** Scikit-Learn

---

## 📁 Recommended Project Structure

```text
AI-Handwritten-Digit-Recognition/
├── app.py                      # Main Streamlit web application
├── train_model.py              # Script to train CNN and MLP models & save evaluation plots
├── preprocessing.py            # Image preprocessing pipeline (MNIST & custom user image)
├── model.py                    # Keras CNN & MLP model builder functions
├── evaluate_model.py           # Evaluation pipeline (confusion matrix & classification report)
├── utils.py                    # Metrics I/O, plotting helper functions, and interactive charts
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
├── models/                     # Saved Keras models
│   ├── mnist_cnn.keras         # Saved CNN model weights
│   └── mnist_mlp.keras         # Saved baseline MLP model weights
├── outputs/                    # Output plots & metrics JSON files
│   ├── metrics.json
│   ├── metrics.txt
│   ├── training_history.json
│   ├── accuracy_curve.png
│   ├── loss_curve.png
│   ├── confusion_matrix.png
│   └── misclassifications.png
└── notebooks/                  # Project report support notebook
    └── model_training.ipynb
```

---

## 🚀 Quick Start & Installation

### 1. Clone or Download Repository
```bash
git clone <repository_url>
cd AI-Handwritten-Digit-Recognition
```

### 2. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model (Optional / Pre-trained Model Included)
To train the CNN model from scratch and generate fresh evaluation plots:
```bash
python train_model.py
```

### 4. Run the Interactive Streamlit Application
```bash
python -m streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 📊 Model Evaluation Results

- **CNN Test Accuracy:** ~99.0%
- **CNN Test Loss:** ~0.03
- **Baseline MLP Test Accuracy:** ~97.5%

---

## 🎓 Academic Viva & Presentation Notes

- **Why CNN over MLP?** CNNs use 2D convolutional kernels to preserve spatial relationships between neighboring pixels, whereas MLPs flatten images into 1D vectors and lose feature geometry.
- **Why Image Preprocessing?** Preprocessing converts raw user drawings or uploaded images into white digits on black backgrounds (28×28), cropped and centered in a 20×20 bounding box to match the statistical distribution of the original MNIST dataset.
- **Softmax Layer:** Converts raw output logits into a normalized 10-class probability distribution summing to 1.0 (100%).

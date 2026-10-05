import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

def load_mnist_data():
    """
    Loads official MNIST handwritten digit dataset from TensorFlow/Keras.
    Returns: (x_train, y_train), (x_test, y_test)
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return (x_train, y_train), (x_test, y_test)

def preprocess_mnist_data(x_train, x_test):
    """
    Preprocesses MNIST train and test sets.
    - Normalizes pixel values from 0-255 to 0.0-1.0 (float32)
    - Reshapes images to (N, 28, 28, 1) for CNN input
    """
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    return x_train, x_test

def preprocess_user_image(image_input):
    """
    Preprocesses custom user input (from drawing canvas or image upload)
    to match the MNIST dataset format expected by the CNN:
    1. Grayscale conversion
    2. Automatic Polarity detection & Inversion (ensuring white digit on black background)
    3. Noise thresholding
    4. Bounding box cropping around the digit
    5. Aspect-ratio preserving resize to fit 20x20 area
    6. Center placement inside 28x28 black canvas
    7. Normalization [0.0, 1.0]
    8. Reshaping to (1, 28, 28, 1)

    Returns:
        model_input: Array of shape (1, 28, 28, 1)
        normalized_28x28: Array of shape (28, 28)
        steps: Dictionary of intermediate PIL Image preprocessing steps for UI visualization
    """
    if isinstance(image_input, np.ndarray):
        # Handle canvas array input (RGBA or RGB or Grayscale)
        if image_input.ndim == 3 and image_input.shape[2] == 4:
            img = Image.fromarray(image_input.astype('uint8'), mode='RGBA')
        elif image_input.ndim == 3 and image_input.shape[2] == 3:
            img = Image.fromarray(image_input.astype('uint8'), mode='RGB')
        else:
            img = Image.fromarray(image_input.astype('uint8'))
    elif isinstance(image_input, Image.Image):
        img = image_input.copy()
    else:
        raise ValueError("Unsupported image input type.")

    # Step 1: Grayscale conversion
    img_gray = img.convert('L')
    img_gray_arr = np.array(img_gray)

    # Step 2: Automatic Polarity Check
    # MNIST digits are white (~255) on black background (~0).
    # Sample border pixels to evaluate background brightness.
    border_pixels = np.concatenate([
        img_gray_arr[0, :], img_gray_arr[-1, :],
        img_gray_arr[:, 0], img_gray_arr[:, -1]
    ])
    bg_is_light = np.mean(border_pixels) > 127
    
    if bg_is_light:
        img_inverted = ImageOps.invert(img_gray)
    else:
        img_inverted = img_gray

    img_arr = np.array(img_inverted)

    # Step 3: Threshold small background noise
    img_arr[img_arr < 25] = 0

    # Step 4: Bounding box crop around non-zero digit pixels
    nonzero_coords = np.argwhere(img_arr > 20)
    
    if nonzero_coords.size > 0:
        min_y, min_x = nonzero_coords.min(axis=0)
        max_y, max_x = nonzero_coords.max(axis=0)
        
        # Add slight padding around the bounding box
        h = max_y - min_y + 1
        w = max_x - min_x + 1
        pad = int(max(h, w) * 0.1) + 2
        
        crop_min_y = max(0, min_y - pad)
        crop_max_y = min(img_arr.shape[0], max_y + pad)
        crop_min_x = max(0, min_x - pad)
        crop_max_x = min(img_arr.shape[1], max_x + pad)
        
        digit_crop_arr = img_arr[crop_min_y:crop_max_y, crop_min_x:crop_max_x]
        cropped_pil = Image.fromarray(digit_crop_arr)
    else:
        cropped_pil = Image.fromarray(img_arr)

    # Step 5: Aspect-ratio preserving resize into 20x20 box
    w, h = cropped_pil.size
    if w > 0 and h > 0:
        if w > h:
            new_w = 20
            new_h = max(1, int(round(20 * (h / float(w)))))
        else:
            new_h = 20
            new_w = max(1, int(round(20 * (w / float(h)))))
        
        resized_digit = cropped_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    else:
        resized_digit = Image.new('L', (20, 20), color=0)

    # Step 6: Center digit on 28x28 black canvas
    canvas = Image.new('L', (28, 28), color=0)
    paste_x = (28 - resized_digit.width) // 2
    paste_y = (28 - resized_digit.height) // 2
    canvas.paste(resized_digit, (paste_x, paste_y))

    # Step 7 & 8: Normalization & reshaping
    final_28x28_arr = np.array(canvas)
    normalized_arr = final_28x28_arr.astype('float32') / 255.0
    model_input = normalized_arr.reshape(1, 28, 28, 1)

    steps = {
        "original": img,
        "grayscale": img_gray,
        "inverted": img_inverted,
        "cropped": cropped_pil,
        "final_28x28": canvas
    }

    return model_input, normalized_arr, steps

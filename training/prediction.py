# For InceptionV3 Trained Models ( For Trial 1 and 2 )

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. CONFIGURATION ---
# This MUST match the input size your model was trained on
IMG_SIZE = 224

# The name of your saved ultimate model file
MODEL_PATH = "MeteorT.keras"

# !!! IMPORTANT: CHANGE THIS PATH to the image you want to test !!!
IMAGE_TO_TEST = 'test_image.jpg'

# Class names must be in alphabetical order as learned by Keras
CLASS_NAMES = ['metal', 'silicate']

# --- 2. LOAD THE TRAINED MODEL ---
print(f"Loading model from: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    print(f"FATAL ERROR: Model file not found at '{MODEL_PATH}'")
    print("Please make sure your trained model file is in the same directory as this script.")
    exit()

# The .keras format loads easily without any special parameters
try:
    best_model = keras.models.load_model(MODEL_PATH)
    print("Ultimate InceptionV3 model loaded successfully.")
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    exit()


# --- 3. PREPARE THE IMAGE ---
print(f"Loading and preparing image: {IMAGE_TO_TEST}")
if not os.path.exists(IMAGE_TO_TEST):
    print(f"FATAL ERROR: Image file not found at '{IMAGE_TO_TEST}'")
    print("Please update the IMAGE_TO_TEST variable with a valid path to your image.")
    exit()

try:
    # Load the image, resizing it to the required 224x224
    img = tf.keras.utils.load_img(
        IMAGE_TO_TEST, target_size=(IMG_SIZE, IMG_SIZE)
    )
    # Convert the image to a NumPy array. This is our raw input.
    img_array = tf.keras.utils.img_to_array(img)

    # Create a "batch" of one image for the model
    # The model will handle all preprocessing (like preprocess_input) internally
    img_array_batch = tf.expand_dims(img_array, 0)

except Exception as e:
    print(f"An error occurred while processing the image: {e}")
    exit()


# --- 4. MAKE PREDICTION ---
print("Making a prediction...")
# The model takes the raw image (0-255 range) and does everything internally
prediction = best_model.predict(img_array_batch)
# The raw output score from the sigmoid function (a value between 0 and 1)
raw_score = prediction[0][0]

# --- 5. INTERPRET THE RESULT AS COMPOSITION ---
# The raw_score is the model's confidence that the image is class 1 ('silicate')
silicate_composition_percentage = raw_score * 100
metal_composition_percentage = (1 - raw_score) * 100

# Determine the final verdict based on which composition is higher
if silicate_composition_percentage > 50:
    final_verdict = "Silicate-rich"
else:
    final_verdict = "Metal-rich"

# --- 6. DISPLAY THE DETAILED RESULT IN THE TERMINAL ---
print("\n" + "="*40)
print("      ULTIMATE MODEL PREDICTION REPORT")
print("="*40)
print(f"Final Verdict:           '{final_verdict}'")
print("-" * 40)
print("Estimated Composition (based on model's confidence):")
print(f"  - Silicate:            {silicate_composition_percentage:.2f}%")
print(f"  - Metal:               {metal_composition_percentage:.2f}%")
print("="*40)
print("(Note: This is the model's confidence based on learned")
print(" textures and patterns, not a literal pixel count.)")


# --- 7. VISUALIZE THE IMAGE WITH THE PREDICTION ---
# Create a title for the plot with the detailed results
title_text = (
    f"Verdict: {final_verdict}\n"
    f"Silicate Confidence: {silicate_composition_percentage:.1f}% | Metal Confidence: {metal_composition_percentage:.1f}%"
)

# Display the image
plt.figure(figsize=(7, 7))
plt.imshow(img)
plt.title(title_text, fontsize=12)
plt.axis("off") # Hide the x and y axes for a cleaner look
plt.show()
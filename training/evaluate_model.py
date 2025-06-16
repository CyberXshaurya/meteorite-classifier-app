import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# --- 1. CONFIGURATION ---
MODEL_PATH = "MeteorT.keras"
DATA_DIR = 'dataset'
IMG_SIZE = 224
CLASS_NAMES = ['metal', 'silicate']

# --- 2. LOAD MODEL AND DATA ---
print(f"Loading model: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    print(f"FATAL ERROR: Model file '{MODEL_PATH}' not found.")
    exit()
model = keras.models.load_model(MODEL_PATH)

# Load all images and labels into memory for a robust, stratified split
print("\nLoading all images into memory for robust splitting...")
all_images = []
all_labels = []
for i, class_name in enumerate(CLASS_NAMES):
    class_dir = os.path.join(DATA_DIR, class_name)
    if not os.path.isdir(class_dir): continue
    for fname in os.listdir(class_dir):
        try:
            img_path = os.path.join(class_dir, fname)
            img = keras.utils.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
            all_images.append(keras.utils.img_to_array(img))
            all_labels.append(i)
        except Exception as e:
            print(f"Could not load image {fname}: {e}")

all_images = np.array(all_images)
all_labels = np.array(all_labels)

# Use scikit-learn for a guaranteed stratified split
X_train, X_val, y_train, y_val = train_test_split(
    all_images, all_labels, test_size=0.2, random_state=123, stratify=all_labels
)
print(f"\nUsing {len(X_val)} images for validation.")
if len(y_val) > 0:
    print(f"Validation set composition: {np.bincount(y_val)[0]} metal, {np.bincount(y_val)[1]} silicate")

# --- 3. PERFORMANCE REPORTS ---
print("\nGenerating predictions on the validation set...")
predictions_prob = model.predict(X_val)
predictions_class = (predictions_prob > 0.5).astype("int32").flatten()

print("\n" + "="*50); print("      CLASSIFICATION REPORT"); print("="*50)
print(classification_report(y_val, predictions_class, target_names=CLASS_NAMES))
print("\n" + "="*50); print("      CONFUSION MATRIX"); print("="*50)
cm = confusion_matrix(y_val, predictions_class)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel('Predicted Label'); plt.ylabel('True Label'); plt.title('Confusion Matrix')
plt.show()

# --- 4. GRAD-CAM HEATMAPS (THE BULLETPROOF FIX) ---
def make_gradcam_heatmap(img_array, grad_model):
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, 0]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = last_conv_layer_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def display_gradcam(img, heatmap):
    img = img.astype(np.uint8)
    heatmap = np.uint8(255 * heatmap)
    jet = plt.cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * 0.4 + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)
    return superimposed_img

# FIX: Find the last convolutional layer programmatically and safely
try:
    base_model = model.get_layer('inception_v3')
    last_conv_layer = None
    for layer in reversed(base_model.layers):
        if isinstance(layer, keras.layers.Conv2D): # Check type, not name or shape
            last_conv_layer = layer
            break
    
    if last_conv_layer:
        print(f"\nUsing layer '{last_conv_layer.name}' for Grad-CAM.")
        grad_model = Model(inputs=model.inputs, outputs=[last_conv_layer.output, model.output])

        print("Generating Grad-CAM Heatmaps...")
        plt.figure(figsize=(15, 10))
        for i in range(min(6, len(X_val))):
            img_array = np.expand_dims(X_val[i], axis=0)
            heatmap = make_gradcam_heatmap(img_array, grad_model)
            superimposed_img = display_gradcam(X_val[i], heatmap)
            plt.subplot(2, 3, i + 1)
            plt.imshow(superimposed_img)
            pred_label = CLASS_NAMES[predictions_class[i]]
            true_label = CLASS_NAMES[y_val[i]]
            plt.title(f"True: {true_label}\nPred: {pred_label}")
            plt.axis('off')
        plt.suptitle("Grad-CAM: What the Model is Focusing On", fontsize=16)
        plt.show()
    else:
        print("Could not programmatically find a suitable convolutional layer for Grad-CAM.")
except Exception as e:
    print(f"Could not generate Grad-CAM heatmaps. Error: {e}")

# --- 5. VISUALIZE FEATURE MAPS (THE BULLETPROOF FIX) ---
print("\nVisualizing Feature Maps from Intermediate Layers...")
try:
    base_model = model.get_layer('inception_v3')
    
    # FIX: Find interesting layers programmatically
    interesting_layer_names = [layer.name for layer in base_model.layers if 'mixed' in layer.name]
    
    if interesting_layer_names:
        layers_to_visualize = [interesting_layer_names[1], # Early-mid layer
                               interesting_layer_names[len(interesting_layer_names)//2], # Mid layer
                               interesting_layer_names[-2]] # Late layer
        
        layer_outputs = [base_model.get_layer(name).output for name in layers_to_visualize]
        feature_map_model = Model(inputs=model.inputs, outputs=layer_outputs)

        img_for_fmap = np.expand_dims(X_val[0], axis=0)
        feature_maps = feature_map_model.predict(img_for_fmap)

        for layer_name, f_map in zip(layers_to_visualize, feature_maps):
            n_features = f_map.shape[-1]
            size = f_map.shape[1]
            display_grid = np.zeros((size, size * min(n_features, 8)))
            for i in range(min(n_features, 8)):
                x = f_map[0, :, :, i]
                x -= x.mean(); x /= (x.std() + 1e-5)
                x *= 64; x += 128
                x = np.clip(x, 0, 255).astype('uint8')
                display_grid[:, i * size : (i + 1) * size] = x
            scale = 20. / 8
            plt.figure(figsize=(scale * 8, scale))
            plt.title(f"Feature Maps from '{layer_name}'")
            plt.grid(False); plt.imshow(display_grid, aspect='auto', cmap='viridis')
            plt.show()
    else:
        print("Could not find 'mixed' layers to visualize feature maps.")
except Exception as e:
    print(f"Could not generate Feature Maps. Error: {e}")

# --- 6. VISUALIZE WEIGHTS & BIASES DISTRIBUTIONS ---
print("\nVisualizing Weights and Biases Distributions for Key Layers...")
for layer in model.layers:
    # We plot for Dense and Conv2D layers
    if isinstance(layer, (keras.layers.Dense, keras.layers.Conv2D)):
        if layer.get_weights():
            weights, biases = layer.get_weights()
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            sns.histplot(weights.flatten(), kde=True)
            plt.title(f"Weights Distribution - {layer.name}")
            plt.subplot(1, 2, 2)
            sns.histplot(biases.flatten(), kde=True)
            plt.title(f"Biases Distribution - {layer.name}")
            plt.tight_layout()
            plt.show()
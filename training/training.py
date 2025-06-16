# InceptionV3 - Trial 2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
import matplotlib.pyplot as plt
import os
import numpy as np

# --- 1. CONFIGURATION ---
IMG_SIZE = 224
BATCH_SIZE = 32
DATA_DIR = 'dataset'
EPOCHS = 120
MIXUP_ALPHA = 0.2

# --- 2. DATA LOADING ---
print("Loading data...")
try:
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="training", seed=123,
        image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, label_mode='binary'
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="validation", seed=123,
        image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, label_mode='binary'
    )
except FileNotFoundError:
    print(f"FATAL ERROR: The directory '{DATA_DIR}' was not found.")
    exit()

class_names = train_ds.class_names
print(f"\nClasses found: {class_names}")

# --- 3. ADVANCED DATA PIPELINE with MIXUP ---
def mixup_data(images, labels):
    alpha = [MIXUP_ALPHA]
    batch_size = tf.shape(images)[0]

    l = tf.compat.v1.distributions.Beta(alpha, alpha).sample(1)
    
    # Reshape lambda for broadcasting
    lambda_images = tf.reshape(l, (1, 1, 1, 1))
    lambda_labels = tf.reshape(l, (1, 1))

    # Shuffle images and labels to mix with
    shuffled_indices = tf.random.shuffle(tf.range(batch_size))
    x_shuffled = tf.gather(images, shuffled_indices)
    y_shuffled = tf.gather(labels, shuffled_indices)

    # Perform mixup
    x_mixed = lambda_images * images + (1 - lambda_images) * x_shuffled
    # Ensure labels are float32 for mixing
    y_mixed = lambda_labels * tf.cast(labels, tf.float32) + (1 - lambda_labels) * tf.cast(y_shuffled, tf.float32)

    return x_mixed, y_mixed

AUTOTUNE = tf.data.AUTOTUNE
# NOTE: We DO NOT apply preprocessing here anymore. It's now inside the model.
# We ONLY apply MixUp to the training data.
train_ds = train_ds.map(mixup_data, num_parallel_calls=AUTOTUNE)

# Configure for performance
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE) # Validation data doesn't need mixup

# --- 4. BUILD THE OPTIMIZED MODEL ---
print("\nBuilding the Optimized Model for Balanced Dataset...")
base_model = InceptionV3(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
base_model.trainable = False

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
# PREPROCESSING IS NOW THE FIRST LAYER OF THE MODEL - This is more robust
x = preprocess_input(inputs)

# The preprocessed input is then passed to the base model
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)

l2_reg = keras.regularizers.l2(1e-5)
x = layers.Dense(1024, kernel_regularizer=l2_reg, activation='gelu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)

x = layers.Dense(512, kernel_regularizer=l2_reg, activation='gelu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)

outputs = layers.Dense(1, activation='sigmoid')(x)
model = Model(inputs=inputs, outputs=outputs)

# --- 5. COMPILE WITH ADVANCED OPTIMIZER ---
optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

# --- 6. ADVANCED CALLBACKS ---
checkpoint_filepath = "balanced_inception_best.keras" # New model name
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath, save_best_only=True, monitor="val_accuracy", mode='max'
)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    patience=20, restore_best_weights=True, monitor='val_accuracy'
)
reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=5, verbose=1, min_lr=1e-7
)

# --- 7. TRAIN THE MODEL (TRANSFER LEARNING) ---
print("\n--- Starting Model Training (Phase 1: Transfer Learning) ---")
# *** CRITICAL CHANGE: class_weight is REMOVED because the dataset is balanced ***
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=[early_stopping_cb, checkpoint_cb, reduce_lr_cb]
)

# --- 8. FINE-TUNING ---
print("\n--- Starting Fine-Tuning (Phase 2) ---")
base_model.trainable = True
for layer in base_model.layers[:249]:
    layer.trainable = False
for layer in base_model.layers[249:]:
    layer.trainable = True

optimizer_ft = tf.keras.optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-4)
model.compile(optimizer=optimizer_ft, loss='binary_crossentropy', metrics=['accuracy'])
print("Re-compiled model for fine-tuning.")

fine_tune_epochs = 30
total_epochs = len(history.epoch) + fine_tune_epochs

history_fine_tune = model.fit(
    train_ds,
    epochs=total_epochs,
    initial_epoch=len(history.epoch),
    validation_data=val_ds,
    callbacks=[early_stopping_cb, checkpoint_cb]
)

# --- 9. EVALUATE AND VISUALIZE ---
print("\n--- Evaluating Final Model ---")
best_model = keras.models.load_model(checkpoint_filepath)
loss, accuracy = best_model.evaluate(val_ds)
print(f"\nFinal Accuracy on Validation Data: {accuracy * 100:.2f}%")

# Combine history for plotting
history.history['accuracy'].extend(history_fine_tune.history.get('accuracy', []))
history.history['val_accuracy'].extend(history_fine_tune.history.get('val_accuracy', []))
history.history['loss'].extend(history_fine_tune.history.get('loss', []))
history.history['val_loss'].extend(history_fine_tune.history.get('val_loss', []))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('balanced_inception_performance.png')
plt.show()
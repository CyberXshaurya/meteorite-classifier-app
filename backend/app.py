import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import base64

# --- CONFIGURATION ---
IMG_SIZE = 224
MODEL_PATH = "MeteorT.keras"
CLASS_NAMES = ['metal', 'silicate']
STATIC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend', 'build'))

# --- INITIALIZE APP and LOAD MODEL ---
app = Flask(__name__, static_folder=STATIC_DIR, static_url_path='/')
CORS(app) # Frontend ko allow karne ke liye

print(f"Loading model from: {MODEL_PATH}")
model = keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

def process_and_predict(image_bytes):
    # YEH FUNCTION BILKUL SAME HAI, KOI CHANGE NAHI
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.utils.img_to_array(img)
    img_array_batch = tf.expand_dims(img_array, 0)
    prediction = model.predict(img_array_batch)
    raw_score = prediction[0][0]
    silicate_percentage = raw_score * 100
    metal_percentage = (1 - raw_score) * 100
    verdict = "Silicate-rich" if raw_score > 0.5 else "Metal-rich"
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return {
        "verdict": verdict,
        "silicate_composition": f"{silicate_percentage:.2f}%",
        "metal_composition": f"{metal_percentage:.2f}%",
        "processed_image": f"data:image/png;base64,{img_base64}"
    }

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        try:
            image_bytes = file.read()
            prediction_result = process_and_predict(image_bytes)
            return jsonify(prediction_result)
        except Exception as e:
            return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/')
def serve():
    return send_from_directory(app.static_folder, 'index.html')

@app.errorhandler(404)
def not_found(e):
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
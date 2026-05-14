from flask import Flask, request, jsonify, render_template
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import tensorflow as tf
import numpy as np
import io
import os

# ================= APP INIT =================
app = Flask(__name__)

# ================= FIRE MODEL =================
FIRE_MODEL_NAME = "prithivMLmods/Forest-Fire-Detection"

print("🔥 Loading Fire model...")
fire_model = AutoModelForImageClassification.from_pretrained(FIRE_MODEL_NAME)
fire_processor = AutoImageProcessor.from_pretrained(FIRE_MODEL_NAME)
print("✅ Fire model loaded")

fire_labels = {
    "0": "Fire",
    "1": "Normal",
    "2": "Smoke"
}

# ================= LEAF MODEL =================
LEAF_MODEL_PATH = os.path.join("models", "leaf_model.h5")

print("🍃 Loading Leaf model...")
leaf_model = tf.keras.models.load_model(LEAF_MODEL_PATH)
print("✅ Leaf model loaded")
# ================= FIRE PREDICTION =================
def predict_fire(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))   # backend resize only

    inputs = fire_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = fire_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0].tolist()

    prediction = {
        fire_labels[str(i)]: round(probs[i], 3)
        for i in range(len(probs))
    }
    top_class = max(prediction, key=prediction.get)

    return top_class, prediction

# ================= LEAF PREDICTION =================
def predict_leaf(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))   # REQUIRED
    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = leaf_model.predict(img, verbose=0)[0][0]

    label = "Green Leaf 🍃" if pred > 0.5 else "Dry Leaf 🍂"
    confidence = round(float(pred), 3)

    return label, confidence

# ================= ROUTES =================
@app.route('/')
def index():
    return render_template("index.html")

# ---------- FIRE ----------
@app.route('/predict', methods=['POST'])
def predict_fire_route():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    top, probs = predict_fire(file.read())
    return jsonify({
        'prediction': top,
        'probabilities': probs
    })

# ---------- LEAF ----------
@app.route('/predict_leaf', methods=['POST'])
def predict_leaf_route():
    if 'leaf_image' not in request.files:
        return jsonify({'error': 'No leaf image uploaded'}), 400

    file = request.files['leaf_image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    label, conf = predict_leaf(file.read())
    return jsonify({
        'prediction': label,
        'confidence': conf
    })

# ================= RUN =================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

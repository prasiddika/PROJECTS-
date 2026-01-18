from flask import Flask, request, render_template, jsonify
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -----------------------------
# Product recommendations
# -----------------------------
product_recommendations = {
    "Oily": [
        "Oil-free moisturizer",
        "Foaming cleanser",
        "Clay face mask"
    ],
    "Dry": [
        "Hydrating moisturizer",
        "Cream cleanser",
        "Hyaluronic acid serum"
    ],
    "Normal": [
        "Light moisturizer",
        "Gentle cleanser",
        "Vitamin C serum"
    ],
    "Combination": [
        "Gel moisturizer",
        "Balancing cleanser",
        "Exfoliating toner"
    ]
}

# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__)

# Load the trained model
model = load_model("model/skin_model.h5")

# -----------------------------
# Image preprocessing
# -----------------------------
def prepare_image(img):
    """Convert uploaded image to MobileNetV2 input"""
    try:
        img = Image.open(img)
        img = img.convert("RGB")          # Ensure 3 channels
        img = img.resize((224, 224))      # MobileNetV2 input size
        img_array = np.array(img)
        img_array = preprocess_input(img_array)   # Normalize pixels
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print("Image preprocessing error:", e)
        return None

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    image_file = request.files.get("image")
    
    if image_file is None:
        return jsonify({"error": "No image uploaded"}), 400

    img_array = prepare_image(image_file)
    if img_array is None:
        return jsonify({"error": "Failed to process image"}), 400

    try:
        # Model prediction
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)

        # Correct class names
        class_names = ["Oily", "Dry", "Normal", "Combination"]
        skin_type = class_names[class_index]

        # Get recommended products
        products = product_recommendations.get(skin_type, [])

        return jsonify({
            "skin_type": skin_type,
            "products": products
        })

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": "Prediction failed"}), 500

# -----------------------------
# Run the app
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)

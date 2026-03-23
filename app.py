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
model = load_model("model/skin_model.keras")

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
    if not image_file:
        return jsonify({"error": "No image"}), 400

    img_array = prepare_image(image_file)
    
    try:
        prediction = model.predict(img_array)
        
        # --- NEW LOGIC FOR RELIABILITY ---
        class_index = np.argmax(prediction)
        confidence_value = np.max(prediction) # This gets the highest probability (0.0 to 1.0)
        
        class_names = ["Oily", "Dry", "Normal", "Combination"]
        skin_type = class_names[class_index]

        # Convert to percentage string for the UI
        confidence_percent = f"{float(confidence_value) * 100:.1f}%"

        return jsonify({
            "skin_type": skin_type,
            "confidence": confidence_percent, # This fixes the "undefined"
            "products": product_recommendations.get(skin_type, [])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# Run the app
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)

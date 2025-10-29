# app.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import io
import base64
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, render_template, request, jsonify
import requests
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB

def create_model():
    """Create the model architecture"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    return model

# Load the trained model.
MODEL_PATH = os.getenv("MODEL_PATH", "wound_model_processed.h5")
MODEL_URL = os.getenv("MODEL_URL")

def ensure_model_file():
    if os.path.exists(MODEL_PATH):
        return
    if not MODEL_URL:
        raise RuntimeError("Model file not found and MODEL_URL is not set.")
    # Stream download to avoid high memory usage
    with requests.get(MODEL_URL, stream=True, timeout=300) as response:
        response.raise_for_status()
        with open(MODEL_PATH, "wb") as model_file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    model_file.write(chunk)
    # Validate file exists and has a reasonable size (>1MB)
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1 * 1024 * 1024:
        raise RuntimeError(
            f"Downloaded model seems invalid or too small at '{MODEL_PATH}'. Check MODEL_URL permissions and link."
        )
try:
    ensure_model_file()
    # Create new model instance
    model = create_model()

    # Load weights
    model.load_weights(MODEL_PATH)

    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

# Define the wound classes (must match the order used during training)
WOUND_CLASSES = [
    'abrasions',
    'bruises',
    'burns',
    'cut',
    'diabetic_wounds',
    'laseration',
    'normal',
    'pressure_wounds',
    'surgical_wounds',
    'venous_wounds'
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/healthz')
def healthz():
    return jsonify({"status": "ok"}), 200

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/upload')
def upload_page():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Expecting a JSON payload with key 'image' containing a base64 encoded image.
        data = request.json
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        img_data = data['image']
        # If the image data has a header, remove it.
        if ',' in img_data:
            header, encoded = img_data.split(',', 1)
        else:
            encoded = img_data
        img_bytes = base64.b64decode(encoded)
        
        # Process the image
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        predictions = model.predict(img_array)
        predicted_index = int(np.argmax(predictions[0]))
        predicted_label = WOUND_CLASSES[predicted_index]
        confidence = float(predictions[0][predicted_index])
        
        return jsonify({'predicted_label': predicted_label, 'confidence': confidence})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload():
    try:
        # Check if the POST request contains a file.
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Only allow jpg/jpeg/png files.
        allowed_extensions = ['jpg', 'jpeg', 'png']
        ext = file.filename.split('.')[-1].lower()
        if ext not in allowed_extensions:
            return jsonify({'error': 'Unsupported file type'}), 400
        
        # Process the uploaded image.
        img = Image.open(file.stream).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array)
        predicted_index = int(np.argmax(predictions[0]))
        predicted_label = WOUND_CLASSES[predicted_index]
        confidence = float(predictions[0][predicted_index])
        
        return jsonify({'predicted_label': predicted_label, 'confidence': confidence})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the Flask development server on http://127.0.0.1:8081
    app.run(debug=True, port=8081)

import os
from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Create upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Update paths to be relative to the current file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'plantdisease.h5')
LABELS_PATH = os.path.join(BASE_DIR, 'labels.json')

# Load the model
model = load_model(MODEL_PATH)

# Load labels
with open(LABELS_PATH, 'r') as f:
    CLASS_NAMES = json.load(f)

# Rest of the code remains the same...
def preprocess_image(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'})

        # Save the uploaded image temporarily
        temp_path = 'temp_image.jpg'
        file.save(temp_path)

        # Preprocess the image
        processed_image = preprocess_image(temp_path)

        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[str(predicted_class_index)]  # Convert index to string if your JSON uses string keys
        confidence = float(np.max(predictions[0]))

        # Remove temporary image
        os.remove(temp_path)

        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True) 
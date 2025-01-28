from http.server import BaseHTTPRequestHandler
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import os
import base64
from io import BytesIO
from PIL import Image

# Load model and labels
model = load_model('model/plantdisease.h5')

# Load labels
with open('model/labels.json', 'r') as f:
    CLASS_NAMES = json.load(f)

def preprocess_image(img):
    # Resize image
    img = img.resize((224, 224))
    
    # Convert to array and preprocess
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    return img_array

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            # Get base64 image
            image_data = data['image'].split(',')[1]
            img = Image.open(BytesIO(base64.b64decode(image_data)))
            
            # Preprocess image
            processed_image = preprocess_image(img)
            
            # Make prediction
            predictions = model.predict(processed_image)
            predicted_class = CLASS_NAMES[str(np.argmax(predictions[0]))]
            confidence = float(np.max(predictions[0]))
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            response = {
                'prediction': predicted_class,
                'confidence': confidence
            }
            
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'status': 'API is running'}).encode()) 
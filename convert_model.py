import tensorflow as tf
import os

# Create model directory if it doesn't exist
if not os.path.exists('model'):
    os.makedirs('model')

# Load the .h5 model
print("Loading model...")
model = tf.keras.models.load_model('plantdisease.h5')

# Convert the model to TFLite format
print("Converting model to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optimize for size (optional)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model
tflite_model = converter.convert()

# Save the TFLite model
print("Saving TFLite model...")
with open('model/plant_disease_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Conversion completed! Model saved as 'model/plant_disease_model.tflite'") 
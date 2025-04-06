from flask import Flask, request, jsonify, render_template
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from deepseek_agent import DeepSeekAgent, create_and_train_model
import sys

# Print information about the environment
print(f"Python version: {sys.version}")
print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")
print(f"Current working directory: {os.getcwd()}")

print("Checking if model exists...")
if not os.path.exists('deepseek_model.h5'):
    print("Model file not found. Creating a new model...")
    create_and_train_model()
    
    # Verify model was created
    if os.path.exists('deepseek_model.h5'):
        print(f"Model created successfully. File size: {os.path.getsize('deepseek_model.h5')} bytes")
    else:
        print("Failed to create model file.")
        exit(1)
else:
    print(f"Model file found. Size: {os.path.getsize('deepseek_model.h5')} bytes")

app = Flask(__name__)

# Initialize agent
try:
    print("Initializing agent...")
    agent = DeepSeekAgent('deepseek_model.h5')
    
    # Test prediction with dummy data
    print("Testing model with dummy prediction...")
    test_data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    test_result = agent.predict(test_data)
    print(f"Test prediction result: {test_result}")
    
    print("Agent initialized and tested successfully!")
except Exception as e:
    print(f"Error initializing agent: {e}")
    print("Trying with SavedModel format...")
    try:
        agent = DeepSeekAgent('deepseek_model_saved')
        print("Agent initialized successfully with SavedModel format!")
    except Exception as e2:
        print(f"Fatal error, could not initialize agent: {e2}")
        exit(1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = [float(x) for x in request.json['input_data'].split(',')]
        if len(input_data) != 10:
            return jsonify({'error': 'Input data must be a comma-separated list of 10 numbers'}), 400

        prediction = agent.predict(input_data)
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
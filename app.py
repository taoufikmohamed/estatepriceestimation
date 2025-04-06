from flask import Flask, request, jsonify, render_template
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow import keras
from deepseek_agent import DeepSeekAgent, create_and_train_model

# Print information about the environment
print(f"Python version: {sys.version}")
print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")
print(f"Current working directory: {os.getcwd()}")

# Check if model exists, create if needed
print("Checking if model exists...")
if not (os.path.exists('deepseek_model') or os.path.exists('deepseek_model.h5')):
    print("Model file not found. Creating a new model...")
    feature_names = create_and_train_model()
    
    # Verify model was created
    if os.path.exists('deepseek_model') or os.path.exists('deepseek_model.h5'):
        print("Model created successfully.")
    else:
        print("Failed to create model file.")
        exit(1)
else:
    print("Model file found.")
    # Get feature names from dataset
    from sklearn.datasets import fetch_california_housing
    feature_names = fetch_california_housing().feature_names

app = Flask(__name__)

# Initialize agent
if os.path.exists('deepseek_model'):
    agent = DeepSeekAgent('deepseek_model')
elif os.path.exists('deepseek_model.h5'):
    agent = DeepSeekAgent('deepseek_model.h5')
else:
    print("No model found. Exiting.")
    exit(1)

@app.route('/')
def index():
    # Pass feature names to the template
    return render_template('index.html', feature_names=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from request
        input_data = request.json.get('input_data')
        print(f"Raw input data: {input_data}")
        
        # Convert to list of floats
        input_values = [float(x) for x in input_data.split(',')]
        print(f"Parsed values: {input_values}")
        print(f"Number of values: {len(input_values)}")
        
        # Check if we have the right number of features
        if len(input_values) != len(feature_names):
            return jsonify({
                'error': f'Input data must be a comma-separated list of {len(feature_names)} numbers'
            }), 400
        
        # Make prediction
        print("Making prediction...")
        prediction = agent.predict(input_values)
        print(f"Raw prediction result: {prediction}")
        
        # Format and return result
        return jsonify({
            'prediction': float(prediction),
            'message': f'Predicted house price: ${float(prediction) * 100000:.2f}'
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
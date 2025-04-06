from flask import Flask, request, jsonify, render_template
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras

class DeepSeekAgent:
    def __init__(self, model_path):
        print(f"Attempting to load model from: {model_path}")
        print(f"File exists check: {os.path.exists(model_path)}")
        
        try:
            self.model = keras.models.load_model(model_path)
            print("Model loaded successfully!")
            print(f"Model summary: {self.model.summary()}")
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            # Initialize model to None if loading fails
            self.model = None
            
            # Try creating a new model as fallback
            try:
                print("Creating new model as fallback...")
                self.model = keras.Sequential([
                    keras.layers.Dense(10, activation='relu', input_shape=(10,)),
                    keras.layers.Dense(1)
                ])
                self.model.compile(optimizer='adam', loss='mse')
                print("Fallback model created successfully!")
            except Exception as e2:
                print(f"Failed to create fallback model: {e2}")

    def predict(self, input_data):
        if self.model is None:
            raise ValueError("Model not loaded correctly. Cannot make predictions.")
        predictions = self.model.predict(np.array([input_data]))
        return predictions[0]

def create_and_train_model():
    model = keras.Sequential([
        keras.layers.Dense(10, activation='relu', input_shape=(10,)),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    x_train = np.random.rand(100, 10)
    y_train = np.random.rand(100, 1)
    model.fit(x_train, y_train, epochs=10)
    model.save('deepseek_model.h5')
    print("Model saved as deepseek_model.h5")

# For direct execution
if __name__ == "__main__":
    if not os.path.exists('deepseek_model.h5'):
        create_and_train_model()
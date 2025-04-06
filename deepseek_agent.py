import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DeepSeekAgent:
    def __init__(self, model_path):
        try:
            print(f"Attempting to load model from: {model_path}")
            # Simple load without branching logic
            self.model = keras.models.load_model(model_path, compile=False)
            # Manually compile with simple options
            self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
            self.scaler = None
            if os.path.exists('scaler.pkl'):
                import pickle
                with open('scaler.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.scaler = None

    def predict(self, input_data):
        if self.model is None:
            raise ValueError("Model not loaded correctly. Cannot make predictions.")
        
        print(f"Input data: {input_data}")
        print(f"Input data type: {type(input_data)}")
        print(f"Input data length: {len(input_data)}")
        
        # Ensure input data has the correct shape
        if len(input_data) != 8:
            raise ValueError(f"Expected 8 features, but got {len(input_data)}. The California Housing dataset requires 8 input values.")
        
        # Convert to numpy array with explicit shape
        input_np = np.array([input_data]).reshape(1, 8)
        print(f"Numpy array shape: {input_np.shape}")
        
        # Standardize if scaler exists
        if self.scaler is not None:
            print("Applying scaler transformation")
            input_np = self.scaler.transform(input_np)
            print(f"Scaled array shape: {input_np.shape}")
        
        # Print shape before prediction
        print(f"Final input shape before prediction: {input_np.shape}")
        
        # Make prediction
        predictions = self.model.predict(input_np, verbose=1)
        print(f"Prediction output: {predictions}")
        print(f"Prediction shape: {predictions.shape}")
        
        return predictions[0]

def create_and_train_model():
    # Load California Housing dataset
    california = fetch_california_housing()
    X, y = california.data, california.target
    feature_names = california.feature_names
    
    print(f"Dataset loaded with {X.shape[1]} features: {feature_names}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler for future use
    import pickle
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Create a deeper model
    print(f"Creating model with input shape ({X.shape[1]},)")
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1)
    ])
    
    # Use string identifiers instead of objects for better serialization
    model.compile(
        optimizer='adam',
        loss='mse',  # Changed from keras.losses.MeanSquaredError()
        metrics=['mae']  # Changed from keras.metrics.MeanAbsoluteError()
    )
    
    # Train model with fewer epochs for faster results
    print("Training model...")
    history = model.fit(
        X_train_scaled, y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_test_scaled, y_test),
        verbose=1
    )
    
    # Evaluate model
    loss, mae = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Model evaluation - Loss: {loss:.4f}, MAE: {mae:.4f}")
    
    # Save model (H5 format first for better compatibility)
    try:
        print("Saving model in H5 format...")
        model.save('deepseek_model.h5')
        print("Model saved successfully in H5 format")
    except Exception as e:
        print(f"Error saving model in H5 format: {e}")
    
    # Test the model with a sample input
    sample_input = X[0:1]  # Take the first sample from the dataset
    sample_input_scaled = scaler.transform(sample_input)
    print(f"Sample input shape: {sample_input.shape}")
    sample_prediction = model.predict(sample_input_scaled, verbose=0)
    print(f"Sample prediction: {sample_prediction[0][0]}")
    print(f"Actual value: {y[0]}")
    
    return feature_names

if __name__ == "__main__":
    if not os.path.exists('deepseek_model.h5'):
        feature_names = create_and_train_model()
        print(f"Model created with these features: {feature_names}")
        print("For prediction, input these features in order.")
    else:
        print("Model already exists. Skipping training.")
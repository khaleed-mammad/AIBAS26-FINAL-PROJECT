import tensorflow as tf
import pandas as pd
import numpy as np
import os

MODEL_PATH = '/Users/khaleed_mammad/Desktop/tmp/knowledgeBase/currentAiSolution.keras'
ACTIVATION_DATA_PATH = '/Users/khaleed_mammad/Desktop/tmp/activationBase/activation_data.csv'

def apply_model():    
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return
    
    # 2. LOAD THE BRAIN
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # 3. LOAD THE TRIGGER DATA
    # We only need the features, so we drop the target column if it exists
    data = pd.read_csv(ACTIVATION_DATA_PATH)
    X = data.drop(columns=['exam_score'], errors='ignore')
    
    # 4. PREDICT
    prediction = model.predict(X, verbose=0)
    
    # 5. OUTPUT RESULT
    result = prediction[0][0]
    print(f"Inference complete.")
    print(f"Input Data Summary: {X.values.tolist()}")
    print(f"PREDICTED EXAM SCORE: {result:.4f}")

if __name__ == "__main__":
    apply_model()
import pandas as pd
import pickle
import os

# 1. DEFINE PATHS (Standardized for the Docker Volume)
MODEL_PATH = '/tmp/knowledgeBase/currentOlsSolution.pkl'
ACTIVATION_DATA_PATH = '/tmp/activationBase/activation_data.csv'

def apply_ols_model():
    
    # 2. LOAD THE BRAIN (.pkl)
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: OLS Model not found at {MODEL_PATH}")
        return
        
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    # 3. LOAD THE TRIGGER DATA
    data = pd.read_csv(ACTIVATION_DATA_PATH)
    # Ensure features match exactly what the OLS was trained on
    X = data.drop(columns=['student_id', 'exam_score'], errors='ignore')
    
    # 4. PREDICT
    prediction = model.predict(X)
    
    # 5. OUTPUT RESULT
    print(f"OLS Inference complete.")
    print(f"PREDICTED EXAM SCORE (OLS): {prediction[0]:.4f}")

if __name__ == "__main__":
    apply_ols_model()
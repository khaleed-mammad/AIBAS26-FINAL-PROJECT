import pandas as pd
import pickle
import os
import statsmodels.api as sm

MODEL_PATH = '/tmp/knowledgeBase/currentOlsSolution.pkl'
ACTIVATION_DATA_PATH = '/tmp/activationBase/activation_data.csv'


def apply_ols_model():
    # 1. LOAD THE BRAIN (.pkl)
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: OLS Model not found at {MODEL_PATH}")
        return
        
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    # 2. LOAD THE TRIGGER DATA
    data = pd.read_csv(ACTIVATION_DATA_PATH)
    
    # Drop target and unnecessary columns
    X = data.drop(columns=['student_id', 'exam_score'], errors='ignore')
    
    X_with_const = sm.add_constant(X, has_constant='add')
    
    # 3. PREDICT
    prediction = model.predict(X_with_const)
    
    # 4. OUTPUT RESULT
    print(f"\n" + "="*30)
    print(f"OLS Inference complete.")
    print(f"PREDICTED EXAM SCORE (OLS): {prediction[0]:.4f}")
    print("="*30)

if __name__ == "__main__":
    apply_ols_model()
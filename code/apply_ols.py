import os
import time
import pandas as pd
import pickle
import statsmodels.api as sm

# Standardized paths
MODEL_PATH = '/tmp/knowledgeBase/currentOlsSolution.pkl'
ACTIVATION_DATA_PATH = '/tmp/activationBase/activation_data.csv'

def apply_ols_model():
    print("\n" + "="*40)
    print("--- OLS SYSTEM ONLINE ---")
    print("="*40)
    
    # 1. WAIT FOR FILES (Robustness Tweak)
    timeout = 60
    files_found = False
    
    while timeout > 0:
        if os.path.exists(MODEL_PATH) and os.path.exists(ACTIVATION_DATA_PATH):
            print("SUCCESS: All required files located!")
            files_found = True
            break
        
        print(f"Waiting for volume sync... ({timeout}s remaining)")
        time.sleep(4)
        timeout -= 4

    if not files_found:
        print("CRITICAL ERROR: Timeout reached. Files not found in volume.")
        return

    try:
        # 2. LOAD THE BRAIN (.pkl)
        print("Loading OLS Model...")
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        
        # 3. LOAD THE TRIGGER DATA
        print("Loading Data...")
        data = pd.read_csv(ACTIVATION_DATA_PATH)
        
        # Drop target and unnecessary columns
        X = data.drop(columns=['student_id', 'exam_score'], errors='ignore')
        
        # --- THE CRITICAL FIX ---
        # Add the constant column so shapes align
        X_with_const = sm.add_constant(X, has_constant='add')
        # ------------------------
        
        # 4. PREDICT
        print("Executing Inference...")
        prediction = model.predict(X_with_const)
        
        # 5. OUTPUT RESULT
        print("\n" + "*"*30)
        print(f"PREDICTED EXAM SCORE (OLS): {prediction[0]:.4f}")
        print("*"*30 + "\n")
        
    except Exception as e:
        print(f"RUNTIME ERROR: {e}")

if __name__ == "__main__":
    apply_ols_model()
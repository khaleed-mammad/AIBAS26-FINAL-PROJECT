import os
import time
import pandas as pd
import tensorflow as tf

# Standardized paths
MODEL_PATH = '/tmp/knowledgeBase/currentAiSolution.keras'
ACTIVATION_DATA_PATH = '/tmp/activationBase/activation_data.csv'

def apply_model():
    print("\n" + "="*40)
    print("--- AI SYSTEM ONLINE ---")
    print("="*40)
    
    # INCREASED TIMEOUT: Copying files takes a split second, but safe is better.
    timeout = 60  
    files_found = False
    
    while timeout > 0:
        # Check what is currently in /tmp
        if os.path.exists('/tmp'):
            print(f"Scanning /tmp... Content: {os.listdir('/tmp')}")
            # Check subfolders content if they exist
            if os.path.exists('/tmp/knowledgeBase'):
                print(f"KB Content: {os.listdir('/tmp/knowledgeBase')}")
        
        if os.path.exists(MODEL_PATH) and os.path.exists(ACTIVATION_DATA_PATH):
            print("SUCCESS: All required files located!")
            files_found = True
            break
        
        print(f"Waiting for volume sync... ({timeout}s remaining)")
        time.sleep(4)
        timeout -= 4

    if not files_found:
        print("CRITICAL ERROR: Timeout reached. Files not found in volume.")
        # Print directory structure for debugging
        for root, dirs, files in os.walk("/tmp"):
            print(root, files)
        return

    try:
        print("Loading Model...")
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        
        print("Loading Data...")
        data = pd.read_csv(ACTIVATION_DATA_PATH)
        
        # Ensure we drop columns that are not features (like ID or Target)
        # Adjust this list based on what your model actually expects!
        X = data.drop(columns=['exam_score', 'student_id'], errors='ignore')
        
        print("Executing Inference...")
        prediction = model.predict(X, verbose=0)
        
        print("\n" + "*"*30)
        # Handle cases where prediction might be an array or single value
        print(f"FINAL RESULT: {prediction[0][0]:.4f}")
        print("*"*30 + "\n")
        
    except Exception as e:
        print(f"RUNTIME ERROR: {e}")

if __name__ == "__main__":
    apply_model()
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# 1. LOAD DATA
# During development, we point to our local files. 
# Later, Docker will map these to /tmp/learningBase/
train_df = pd.read_csv('data/data_cleaning/training_data.csv')
test_df = pd.read_csv('data/data_cleaning/test_data.csv')

# Drop student_id (not a feature) and separate Target (exam_score)
X_train = train_df.drop(columns=['exam_score'])
y_train = train_df['exam_score']
X_test = test_df.drop(columns=['exam_score'])
y_test = test_df['exam_score']

# 2. DEFINE THE ANN (The "Wiring")
model = tf.keras.Sequential([
    # Input layer: neurons equal to number of features
    layers.Dense(64, activation='relu', input_shape=[X_train.shape[1]]),
    # Hidden layer: learns complex patterns
    layers.Dense(32, activation='relu'),
    # Output layer: 1 node for the predicted score
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 3. TRAINING
# "epochs" is how many times the AI reads the data. 
# We store results in 'history' to make the plots.
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    verbose=1
)

# 4. SAVE THE MODEL (Required for Subgoal 6)
model.save('currentAiSolution.h5')
print("Model saved as currentAiSolution.h5")
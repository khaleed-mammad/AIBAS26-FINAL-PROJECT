
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
from datetime import datetime
import seaborn as sns
from scipy import stats

# 1. LOAD DATA
# During development, we point to our local files. 
# Later, Docker will map these to /tmp/learningBase/
train_df = pd.read_csv('../data/data_cleaning/training_data.csv')
test_df = pd.read_csv('../data/data_cleaning/test_data.csv')

# Drop student_id (not a feature) and separate Target (exam_score)
X_train = train_df.drop(columns=[ 'exam_score'])
y_train = train_df['exam_score']
X_test = test_df.drop(columns=['exam_score'])
y_test = test_df['exam_score']

# 2. DEFINE THE ANN (The "Wiring")
model = tf.keras.Sequential([
    # Input layer: more neurons for better feature learning
    layers.Dense(128, activation='relu', input_shape=[X_train.shape[1]]),
    layers.BatchNormalization(),  # Stabilizes learning
    layers.Dropout(0.3),  # Prevents overfitting
    
    # Deep hidden layers: learns increasingly complex patterns
    layers.Dense(96, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    
    # Output layer: 1 node for the predicted score
    layers.Dense(1)
])

# Using a learning rate scheduler for better convergence
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

# 3. TRAINING
# "epochs" is how many times the AI reads the data. 
# We store results in 'history' to make the plots.
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=15, 
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# 4. SAVE THE MODEL (Required for Subgoal 6)
model.save('currentAiSolution.h5')
print("Model saved as currentAiSolution.h5")

# 5. GENERATE PREDICTIONS FOR EXAMPLES
y_train_pred = model.predict(X_train, verbose=0)
y_test_pred = model.predict(X_test, verbose=0)

# 6. DOCUMENTATION AND RESULTS
# Create learningBase directory if it doesn't exist (for Docker compatibility)
output_dir = '../images/learningBase_ExamScorePrediction'
os.makedirs(output_dir, exist_ok=True)

# Write comprehensive training report
report_path = os.path.join(output_dir, 'training_report.txt')
with open(report_path, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("EXAM SCORE PREDICTION - ARTIFICIAL NEURAL NETWORK\n")
    f.write("Training Report\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # Model Architecture
    f.write("-" * 80 + "\n")
    f.write("MODEL ARCHITECTURE\n")
    f.write("-" * 80 + "\n")
    f.write(f"Model Type: Sequential Artificial Neural Network (ANN)\n")
    f.write(f"Input Features: {X_train.shape[1]}\n")
    f.write(f"Total Parameters: {model.count_params():,}\n\n")
    
    f.write("Layer Configuration:\n")
    f.write("  Layer 1: Dense(128, activation='relu') + BatchNormalization + Dropout(0.3)\n")
    f.write("  Layer 2: Dense(96, activation='relu') + BatchNormalization + Dropout(0.3)\n")
    f.write("  Layer 3: Dense(64, activation='relu') + BatchNormalization + Dropout(0.2)\n")
    f.write("  Layer 4: Dense(32, activation='relu') + Dropout(0.2)\n")
    f.write("  Output Layer: Dense(1, activation='linear')\n\n")
    
    # Hyperparameters
    f.write("-" * 80 + "\n")
    f.write("HYPERPARAMETERS\n")
    f.write("-" * 80 + "\n")
    f.write(f"Optimizer: Adam\n")
    f.write(f"Learning Rate: 0.001\n")
    f.write(f"Loss Function: Mean Squared Error (MSE)\n")
    f.write(f"Metrics: Mean Absolute Error (MAE), MSE\n")
    f.write(f"Batch Size: 32\n")
    f.write(f"Maximum Epochs: 100\n")
    f.write(f"Early Stopping: Enabled (patience=15, monitor='val_loss')\n")
    f.write(f"Training Samples: {len(X_train)}\n")
    f.write(f"Validation Samples: {len(X_test)}\n\n")
    
    # Training Results
    f.write("-" * 80 + "\n")
    f.write("TRAINING RESULTS\n")
    f.write("-" * 80 + "\n")
    f.write(f"Total Epochs Trained: {len(history.history['loss'])}\n")
    f.write(f"Final Training Loss (MSE): {history.history['loss'][-1]:.4f}\n")
    f.write(f"Final Training MAE: {history.history['mae'][-1]:.4f}\n")
    f.write(f"Final Validation Loss (MSE): {history.history['val_loss'][-1]:.4f}\n")
    f.write(f"Final Validation MAE: {history.history['val_mae'][-1]:.4f}\n\n")
    
    f.write(f"Best Validation Loss (MSE): {min(history.history['val_loss']):.4f}\n")
    f.write(f"Best Validation MAE: {min(history.history['val_mae']):.4f}\n")
    f.write(f"Best Epoch: {np.argmin(history.history['val_loss']) + 1}\n\n")
    
    # Epoch-by-Epoch Results
    f.write("-" * 80 + "\n")
    f.write("EPOCH-BY-EPOCH TRAINING HISTORY\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'Epoch':<8} {'Train Loss':<12} {'Train MAE':<12} {'Val Loss':<12} {'Val MAE':<12}\n")
    f.write("-" * 80 + "\n")
    
    for epoch in range(len(history.history['loss'])):
        f.write(f"{epoch+1:<8} "
                f"{history.history['loss'][epoch]:<12.4f} "
                f"{history.history['mae'][epoch]:<12.4f} "
                f"{history.history['val_loss'][epoch]:<12.4f} "
                f"{history.history['val_mae'][epoch]:<12.4f}\n")
    
    # Example Predictions
    f.write("\n" + "-" * 80 + "\n")
    f.write("EXAMPLE PREDICTIONS\n")
    f.write("-" * 80 + "\n\n")
    
    # Training example
    train_idx = np.random.randint(0, len(y_train))
    f.write(f"Training Set Example (Index {train_idx}):\n")
    f.write(f"  Actual Exam Score: {y_train.iloc[train_idx]:.2f}\n")
    f.write(f"  Predicted Exam Score: {y_train_pred[train_idx][0]:.2f}\n")
    f.write(f"  Prediction Error: {abs(y_train.iloc[train_idx] - y_train_pred[train_idx][0]):.2f}\n\n")
    
    # Test example
    test_idx = np.random.randint(0, len(y_test))
    f.write(f"Test Set Example (Index {test_idx}):\n")
    f.write(f"  Actual Exam Score: {y_test.iloc[test_idx]:.2f}\n")
    f.write(f"  Predicted Exam Score: {y_test_pred[test_idx][0]:.2f}\n")
    f.write(f"  Prediction Error: {abs(y_test.iloc[test_idx] - y_test_pred[test_idx][0]):.2f}\n\n")
    
    # Additional examples
    f.write("Additional Test Set Examples (First 10):\n")
    f.write(f"{'Index':<8} {'Actual':<12} {'Predicted':<12} {'Error':<12}\n")
    f.write("-" * 50 + "\n")
    for i in range(min(10, len(y_test))):
        error = abs(y_test.iloc[i] - y_test_pred[i][0])
        f.write(f"{i:<8} {y_test.iloc[i]:<12.2f} {y_test_pred[i][0]:<12.2f} {error:<12.2f}\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("END OF REPORT\n")
    f.write("=" * 80 + "\n")

print(f"Training report saved to {report_path}")

# 7. VISUALIZATIONS
# (1) Training and Testing Curves
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Model Training and Validation Performance', fontsize=16, fontweight='bold')

# Loss curves
axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0, 0].set_xlabel('Epoch', fontsize=12)
axes[0, 0].set_ylabel('Mean Squared Error', fontsize=12)
axes[0, 0].set_title('Training and Validation Loss (MSE)', fontsize=14)
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# MAE curves
axes[0, 1].plot(history.history['mae'], label='Training MAE', linewidth=2)
axes[0, 1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
axes[0, 1].set_xlabel('Epoch', fontsize=12)
axes[0, 1].set_ylabel('Mean Absolute Error', fontsize=12)
axes[0, 1].set_title('Training and Validation MAE', fontsize=14)
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3)

# Training vs Validation comparison at final epoch
metrics_comparison = ['Loss (MSE)', 'MAE']
train_values = [history.history['loss'][-1], history.history['mae'][-1]]
val_values = [history.history['val_loss'][-1], history.history['val_mae'][-1]]

x_pos = np.arange(len(metrics_comparison))
width = 0.35

axes[1, 0].bar(x_pos - width/2, train_values, width, label='Training', alpha=0.8)
axes[1, 0].bar(x_pos + width/2, val_values, width, label='Validation', alpha=0.8)
axes[1, 0].set_xlabel('Metric', fontsize=12)
axes[1, 0].set_ylabel('Value', fontsize=12)
axes[1, 0].set_title('Final Performance Comparison', fontsize=14)
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(metrics_comparison)
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Summary statistics
axes[1, 1].axis('off')
summary_text = f"""
Final Training Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Epochs Trained: {len(history.history['loss'])}
Best Epoch: {np.argmin(history.history['val_loss']) + 1}

Training Set:
  • Loss (MSE): {history.history['loss'][-1]:.2f}
  • MAE: {history.history['mae'][-1]:.2f}

Validation Set:
  • Loss (MSE): {history.history['val_loss'][-1]:.2f}
  • MAE: {history.history['val_mae'][-1]:.2f}

Best Validation:
  • Loss (MSE): {min(history.history['val_loss']):.2f}
  • MAE: {min(history.history['val_mae']):.2f}
"""
axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, 
                verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'training_validation_curves.pdf'), dpi=300, bbox_inches='tight')
print(f"Training curves saved to {os.path.join(output_dir, 'training_validation_curves.pdf')}")
plt.close()

# (2) Diagnostic Plots - Inspired by Linear Regression Diagnostics
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Regression Diagnostic Plots (Test Set)', fontsize=16, fontweight='bold')

residuals_test = y_test.values - y_test_pred.flatten()
standardized_residuals = residuals_test / np.std(residuals_test)

# Plot 1: Residuals vs Fitted (like in linear regression diagnostics)
axes[0, 0].scatter(y_test_pred, residuals_test, alpha=0.5, edgecolors='k', linewidth=0.5)
axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
# Add lowess smoothing line
sns.regplot(x=y_test_pred.flatten(), y=residuals_test, scatter=False, 
            lowess=True, line_kws={'color': 'red', 'lw': 2, 'alpha': 0.8}, ax=axes[0, 0])
# Annotate top 3 residuals
abs_resid = np.argsort(np.abs(residuals_test))[::-1][:3]
for i in abs_resid:
    axes[0, 0].annotate(f'{i}', xy=(y_test_pred[i], residuals_test[i]), 
                       color='C3', fontsize=9)
axes[0, 0].set_xlabel('Fitted Values (Predicted Scores)', fontsize=12)
axes[0, 0].set_ylabel('Residuals', fontsize=12)
axes[0, 0].set_title('Residuals vs Fitted', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Normal Q-Q Plot (check normality of residuals)
stats.probplot(standardized_residuals, dist="norm", plot=axes[0, 1])
axes[0, 1].get_lines()[0].set_markerfacecolor('C0')
axes[0, 1].get_lines()[0].set_markeredgecolor('k')
axes[0, 1].get_lines()[0].set_alpha(0.5)
# Annotate top 3 residuals
abs_std_resid = np.argsort(np.abs(standardized_residuals))[::-1][:3]
theoretical_quantiles = stats.norm.ppf((np.arange(len(standardized_residuals)) + 0.5) / len(standardized_residuals))
sorted_resid = np.sort(standardized_residuals)
for idx in abs_std_resid:
    # Find position in sorted array
    pos = np.where(np.sort(standardized_residuals) == standardized_residuals[idx])[0][0]
    axes[0, 1].annotate(f'{idx}', xy=(theoretical_quantiles[pos], sorted_resid[pos]), 
                       color='C3', fontsize=9, ha='right')
axes[0, 1].set_title('Normal Q-Q', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Theoretical Quantiles', fontsize=12)
axes[0, 1].set_ylabel('Standardized Residuals', fontsize=12)
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Scale-Location Plot (check homoscedasticity)
sqrt_abs_std_resid = np.sqrt(np.abs(standardized_residuals))
axes[1, 0].scatter(y_test_pred, sqrt_abs_std_resid, alpha=0.5, edgecolors='k', linewidth=0.5)
sns.regplot(x=y_test_pred.flatten(), y=sqrt_abs_std_resid, scatter=False, 
            lowess=True, line_kws={'color': 'red', 'lw': 2, 'alpha': 0.8}, ax=axes[1, 0])
# Annotate top 3
abs_sqrt_resid = np.argsort(sqrt_abs_std_resid)[::-1][:3]
for i in abs_sqrt_resid:
    axes[1, 0].annotate(f'{i}', xy=(y_test_pred[i], sqrt_abs_std_resid[i]), 
                       color='C3', fontsize=9)
axes[1, 0].set_xlabel('Fitted Values (Predicted Scores)', fontsize=12)
axes[1, 0].set_ylabel('√|Standardized Residuals|', fontsize=12)
axes[1, 0].set_title('Scale-Location', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Residual Histogram with Normal Curve
axes[1, 1].hist(standardized_residuals, bins=30, density=True, 
                alpha=0.7, edgecolor='black', label='Residuals')
# Fit normal distribution
mu, sigma = np.mean(standardized_residuals), np.std(standardized_residuals)
x_norm = np.linspace(standardized_residuals.min(), standardized_residuals.max(), 100)
axes[1, 1].plot(x_norm, stats.norm.pdf(x_norm, mu, sigma), 
                'r-', linewidth=2, label=f'Normal(μ={mu:.2f}, σ={sigma:.2f})')
axes[1, 1].axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
axes[1, 1].set_xlabel('Standardized Residuals', fontsize=12)
axes[1, 1].set_ylabel('Density', fontsize=12)
axes[1, 1].set_title('Residual Distribution', fontsize=14, fontweight='bold')
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'diagnostic_plots.pdf'), dpi=300, bbox_inches='tight')
print(f"Diagnostic plots saved to {os.path.join(output_dir, 'diagnostic_plots.pdf')}")
plt.close()

# (3) Scatter Plots - Predicted vs Actual
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Predicted vs Actual Exam Scores', fontsize=16, fontweight='bold')

# Training set
axes[0].scatter(y_train, y_train_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
min_val = min(y_train.min(), y_train_pred.min())
max_val = max(y_train.max(), y_train_pred.max())
axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Exam Score', fontsize=12)
axes[0].set_ylabel('Predicted Exam Score', fontsize=12)
axes[0].set_title(f'Training Set (n={len(y_train)})', fontsize=14)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_aspect('equal', adjustable='box')
# Add R² annotation
train_r2 = 1 - (np.sum((y_train.values - y_train_pred.flatten())**2) / 
                np.sum((y_train.values - y_train.mean())**2))
axes[0].text(0.05, 0.95, f'R² = {train_r2:.4f}', transform=axes[0].transAxes, 
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Test set
axes[1].scatter(y_test, y_test_pred, alpha=0.5, edgecolors='k', linewidth=0.5, color='orange')
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
axes[1].set_xlabel('Actual Exam Score', fontsize=12)
axes[1].set_ylabel('Predicted Exam Score', fontsize=12)
axes[1].set_title(f'Test Set (n={len(y_test)})', fontsize=14)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_aspect('equal', adjustable='box')
# Add R² annotation
test_r2 = 1 - (np.sum((y_test.values - y_test_pred.flatten())**2) / 
               np.sum((y_test.values - y_test.mean())**2))
axes[1].text(0.05, 0.95, f'R² = {test_r2:.4f}', transform=axes[1].transAxes, 
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'scatter_plots.pdf'), dpi=300, bbox_inches='tight')
print(f"Scatter plots saved to {os.path.join(output_dir, 'scatter_plots.pdf')}")
plt.close()

print("\n" + "="*80)
print("All visualizations and documentation generated successfully!")
print(f"Check the '{output_dir}' directory for all outputs.")
print("="*80)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pickle

# 1. LOAD DATA
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

# 1. LOAD DATA
train_path = os.path.join(ROOT_DIR, 'data', 'data_cleaning', 'training_data.csv')
test_path = os.path.join(ROOT_DIR, 'data', 'data_cleaning', 'test_data.csv')

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Drop student_id (not a feature) and separate Target (exam_score)
X_train = train_df.drop(columns=['exam_score'])
y_train = train_df['exam_score']
X_test = test_df.drop(columns=['exam_score'])
y_test = test_df['exam_score']

# 2. DEFINE THE OLS MODEL
# Add constant term for intercept
X_train_with_const = sm.add_constant(X_train)
X_test_with_const = sm.add_constant(X_test)

# Fit OLS model
print("Training OLS Model...")
model = sm.OLS(y_train, X_train_with_const)
results = model.fit()

print("\n" + "="*80)
print("OLS MODEL SUMMARY")
print("="*80)
print(results.summary())
print("="*80 + "\n")

# 3. SAVE THE MODEL
model_save_path = os.path.join(ROOT_DIR, 'currentOlsSolution.pkl')
with open(model_save_path, 'wb') as f:
    pickle.dump(results, f)

# 4. GENERATE PREDICTIONS
y_train_pred = results.predict(X_train_with_const)
y_test_pred = results.predict(X_test_with_const)

# Calculate performance metrics
train_mse = np.mean((y_train - y_train_pred)**2)
train_mae = np.mean(np.abs(y_train - y_train_pred))
train_r2 = results.rsquared

test_mse = np.mean((y_test - y_test_pred)**2)
test_mae = np.mean(np.abs(y_test - y_test_pred))
test_r2 = 1 - (np.sum((y_test - y_test_pred)**2) / np.sum((y_test - y_test.mean())**2))

# 5. DOCUMENTATION AND RESULTS
output_dir = os.path.join(ROOT_DIR, 'images', 'learningBase_ExamScorePrediction')
os.makedirs(output_dir, exist_ok=True)

# Write comprehensive training report
report_path = os.path.join(output_dir, 'ols_training_report.txt')
with open(report_path, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("EXAM SCORE PREDICTION - ORDINARY LEAST SQUARES (OLS) REGRESSION\n")
    f.write("Training Report\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # Model Architecture
    f.write("-" * 80 + "\n")
    f.write("MODEL ARCHITECTURE\n")
    f.write("-" * 80 + "\n")
    f.write(f"Model Type: Ordinary Least Squares (OLS) Linear Regression\n")
    f.write(f"Input Features: {X_train.shape[1]}\n")
    f.write(f"Total Parameters: {len(results.params)}\n")
    f.write(f"Estimation Method: Least Squares\n\n")
    
    # Model Coefficients
    f.write("Feature Coefficients:\n")
    for i, (feature, coef) in enumerate(zip(results.params.index, results.params.values)):
        p_value = results.pvalues[i]
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        f.write(f"  {feature:<30s}: {coef:>12.4f}  (p={p_value:.4f}) {significance}\n")
    
    f.write("\nSignificance codes: *** p<0.001, ** p<0.01, * p<0.05\n\n")
    
    # Model Statistics
    f.write("-" * 80 + "\n")
    f.write("MODEL STATISTICS\n")
    f.write("-" * 80 + "\n")
    f.write(f"Training Samples: {len(X_train)}\n")
    f.write(f"Validation Samples: {len(X_test)}\n")
    f.write(f"Degrees of Freedom (Residuals): {results.df_resid}\n")
    f.write(f"Degrees of Freedom (Model): {results.df_model}\n\n")
    
    f.write(f"R-squared: {results.rsquared:.4f}\n")
    f.write(f"Adjusted R-squared: {results.rsquared_adj:.4f}\n")
    f.write(f"F-statistic: {results.fvalue:.4f}\n")
    f.write(f"Prob (F-statistic): {results.f_pvalue:.4e}\n")
    f.write(f"AIC: {results.aic:.4f}\n")
    f.write(f"BIC: {results.bic:.4f}\n\n")
    
    # Training Results
    f.write("-" * 80 + "\n")
    f.write("TRAINING RESULTS\n")
    f.write("-" * 80 + "\n")
    f.write(f"Training MSE: {train_mse:.4f}\n")
    f.write(f"Training MAE: {train_mae:.4f}\n")
    f.write(f"Training R²: {train_r2:.4f}\n\n")
    
    f.write(f"Validation MSE: {test_mse:.4f}\n")
    f.write(f"Validation MAE: {test_mae:.4f}\n")
    f.write(f"Validation R²: {test_r2:.4f}\n\n")
    
    # VIF (Variance Inflation Factor) - Check for multicollinearity
    f.write("-" * 80 + "\n")
    f.write("VARIANCE INFLATION FACTOR (VIF) - MULTICOLLINEARITY CHECK\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'Feature':<30s} {'VIF':>12s}\n")
    f.write("-" * 80 + "\n")
    
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_train_with_const.columns
    vif_data["VIF"] = [variance_inflation_factor(X_train_with_const.values, i) 
                       for i in range(X_train_with_const.shape[1])]
    
    for _, row in vif_data.iterrows():
        warning = " (High)" if row['VIF'] > 5 else ""
        f.write(f"{row['Feature']:<30s} {row['VIF']:>12.2f}{warning}\n")
    
    f.write("\nNote: VIF > 5 indicates potential multicollinearity issues\n\n")
    
    # Example Predictions
    f.write("-" * 80 + "\n")
    f.write("EXAMPLE PREDICTIONS\n")
    f.write("-" * 80 + "\n\n")
    
    # Training example
    train_idx = np.random.randint(0, len(y_train))
    f.write(f"Training Set Example (Index {train_idx}):\n")
    f.write(f"  Actual Exam Score: {y_train.iloc[train_idx]:.2f}\n")
    f.write(f"  Predicted Exam Score: {y_train_pred.iloc[train_idx]:.2f}\n")
    f.write(f"  Prediction Error: {abs(y_train.iloc[train_idx] - y_train_pred.iloc[train_idx]):.2f}\n\n")
    
    # Test example
    test_idx = np.random.randint(0, len(y_test))
    f.write(f"Test Set Example (Index {test_idx}):\n")
    f.write(f"  Actual Exam Score: {y_test.iloc[test_idx]:.2f}\n")
    f.write(f"  Predicted Exam Score: {y_test_pred.iloc[test_idx]:.2f}\n")
    f.write(f"  Prediction Error: {abs(y_test.iloc[test_idx] - y_test_pred.iloc[test_idx]):.2f}\n\n")
    
    # Additional examples
    f.write("Additional Test Set Examples (First 10):\n")
    f.write(f"{'Index':<8} {'Actual':<12} {'Predicted':<12} {'Error':<12}\n")
    f.write("-" * 50 + "\n")
    for i in range(min(10, len(y_test))):
        error = abs(y_test.iloc[i] - y_test_pred.iloc[i])
        f.write(f"{i:<8} {y_test.iloc[i]:<12.2f} {y_test_pred.iloc[i]:<12.2f} {error:<12.2f}\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("END OF REPORT\n")
    f.write("=" * 80 + "\n")

print(f"OLS training report saved to {report_path}")

# 6. DIAGNOSTIC PLOTS - Using statsmodels approach
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('OLS Regression Diagnostic Plots (Test Set)', fontsize=16, fontweight='bold')

residuals_test = y_test.values - y_test_pred.values
standardized_residuals = residuals_test / np.std(residuals_test)
influence = results.get_influence()
residuals_train = results.resid
standardized_residuals_train = influence.resid_studentized_internal

# Plot 1: Residuals vs Fitted
axes[0, 0].scatter(y_test_pred, residuals_test, alpha=0.5, edgecolors='k', linewidth=0.5)
axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
# Add smooth line
z = np.polyfit(y_test_pred, residuals_test, 2)
p = np.poly1d(z)
x_smooth = np.linspace(y_test_pred.min(), y_test_pred.max(), 100)
axes[0, 0].plot(x_smooth, p(x_smooth), 'r-', linewidth=2, alpha=0.8)
# Annotate top 3 residuals
abs_resid = np.argsort(np.abs(residuals_test))[::-1][:3]
for i in abs_resid:
    axes[0, 0].annotate(f'{i}', xy=(y_test_pred.iloc[i], residuals_test[i]), 
                       color='C3', fontsize=9)
axes[0, 0].set_xlabel('Fitted Values (Predicted Scores)', fontsize=12)
axes[0, 0].set_ylabel('Residuals', fontsize=12)
axes[0, 0].set_title('Residuals vs Fitted', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Normal Q-Q Plot
stats.probplot(standardized_residuals, dist="norm", plot=axes[0, 1])
axes[0, 1].get_lines()[0].set_markerfacecolor('C0')
axes[0, 1].get_lines()[0].set_markeredgecolor('k')
axes[0, 1].get_lines()[0].set_alpha(0.5)
# Annotate top 3 residuals
abs_std_resid = np.argsort(np.abs(standardized_residuals))[::-1][:3]
theoretical_quantiles = stats.norm.ppf((np.arange(len(standardized_residuals)) + 0.5) / len(standardized_residuals))
sorted_resid = np.sort(standardized_residuals)
for idx in abs_std_resid:
    pos = np.where(np.sort(standardized_residuals) == standardized_residuals[idx])[0][0]
    axes[0, 1].annotate(f'{idx}', xy=(theoretical_quantiles[pos], sorted_resid[pos]), 
                       color='C3', fontsize=9, ha='right')
axes[0, 1].set_title('Normal Q-Q', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Theoretical Quantiles', fontsize=12)
axes[0, 1].set_ylabel('Standardized Residuals', fontsize=12)
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Scale-Location Plot
sqrt_abs_std_resid = np.sqrt(np.abs(standardized_residuals))
axes[1, 0].scatter(y_test_pred, sqrt_abs_std_resid, alpha=0.5, edgecolors='k', linewidth=0.5)
# Add smooth line
z = np.polyfit(y_test_pred, sqrt_abs_std_resid, 2)
p = np.poly1d(z)
axes[1, 0].plot(x_smooth, p(x_smooth), 'r-', linewidth=2, alpha=0.8)
# Annotate top 3
abs_sqrt_resid = np.argsort(sqrt_abs_std_resid)[::-1][:3]
for i in abs_sqrt_resid:
    axes[1, 0].annotate(f'{i}', xy=(y_test_pred.iloc[i], sqrt_abs_std_resid[i]), 
                       color='C3', fontsize=9)
axes[1, 0].set_xlabel('Fitted Values (Predicted Scores)', fontsize=12)
axes[1, 0].set_ylabel('√|Standardized Residuals|', fontsize=12)
axes[1, 0].set_title('Scale-Location', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Residuals vs Leverage (using training data as we have influence measures)
leverage = influence.hat_matrix_diag
cooks_d = influence.cooks_distance[0]
axes[1, 1].scatter(leverage, standardized_residuals_train, alpha=0.5, edgecolors='k', linewidth=0.5)
axes[1, 1].axhline(y=0, ls='dotted', color='black', lw=1.25)

# Add Cook's distance contours
leverage_top_3 = np.argsort(cooks_d)[::-1][:3]
for i in leverage_top_3:
    axes[1, 1].annotate(f'{i}', xy=(leverage[i], standardized_residuals_train[i]), 
                       color='C3', fontsize=9)

# Cook's distance threshold lines
n = len(leverage)
p = len(results.params)
x_line = np.linspace(0.001, max(leverage), 50)
for factor in [0.5, 1]:
    y_line = np.sqrt((factor * p * (1 - x_line)) / x_line)
    label = "Cook's distance" if factor == 1 else None
    axes[1, 1].plot(x_line, y_line, ls='--', color='red', lw=1.25, label=label)
    axes[1, 1].plot(x_line, -y_line, ls='--', color='red', lw=1.25)

axes[1, 1].set_xlabel('Leverage', fontsize=12)
axes[1, 1].set_ylabel('Standardized Residuals', fontsize=12)
axes[1, 1].set_title('Residuals vs Leverage', fontsize=14, fontweight='bold')
axes[1, 1].legend(loc='best', fontsize=9)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'ols_diagnostic_plots.png'), dpi=300, bbox_inches='tight')
print(f"OLS diagnostic plots saved to {os.path.join(output_dir, 'ols_diagnostic_plots.png')}")
plt.close()

# 7. SCATTER PLOTS - Predicted vs Actual
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('OLS: Predicted vs Actual Exam Scores', fontsize=16, fontweight='bold')

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
axes[0].text(0.05, 0.95, f'R² = {train_r2:.4f}\nMAE = {train_mae:.2f}', 
            transform=axes[0].transAxes, fontsize=11, verticalalignment='top',
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
axes[1].text(0.05, 0.95, f'R² = {test_r2:.4f}\nMAE = {test_mae:.2f}', 
            transform=axes[1].transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'ols_scatter_plots.png'), dpi=300, bbox_inches='tight')
print(f"OLS scatter plots saved to {os.path.join(output_dir, 'ols_scatter_plots.png')}")
plt.close()

# 8. PERFORMANCE COMPARISON VISUALIZATION
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('OLS Model Performance Summary', fontsize=16, fontweight='bold')

# Performance metrics comparison
metrics = ['MSE', 'MAE', 'R²']
train_vals = [train_mse, train_mae, train_r2]
test_vals = [test_mse, test_mae, test_r2]

x_pos = np.arange(len(metrics))
width = 0.35

axes[0].bar(x_pos - width/2, train_vals, width, label='Training', alpha=0.8)
axes[0].bar(x_pos + width/2, test_vals, width, label='Validation', alpha=0.8)
axes[0].set_xlabel('Metric', fontsize=12)
axes[0].set_ylabel('Value', fontsize=12)
axes[0].set_title('Performance Metrics Comparison', fontsize=14)
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(metrics)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3, axis='y')

# Model summary statistics
axes[1].axis('off')
summary_text = f"""
OLS Model Summary:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Training Results:
  • MSE: {train_mse:.2f}
  • MAE: {train_mae:.2f}
  • R²: {train_r2:.4f}

Validation Results:
  • MSE: {test_mse:.2f}
  • MAE: {test_mae:.2f}
  • R²: {test_r2:.4f}

Model Statistics:
  • Adj. R²: {results.rsquared_adj:.4f}
  • AIC: {results.aic:.2f}
  • BIC: {results.bic:.2f}
  • F-statistic: {results.fvalue:.2f}
  • Parameters: {len(results.params)}
"""
axes[1].text(0.1, 0.5, summary_text, fontsize=11, 
            verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'ols_performance_summary.png'), dpi=300, bbox_inches='tight')
print(f"OLS performance summary saved to {os.path.join(output_dir, 'ols_performance_summary.png')}")
plt.close()

print("\n" + "="*80)
print("OLS MODEL TRAINING COMPLETE!")
print("="*80)
print(f"\nPerformance Summary:")
print(f"  Training   - MSE: {train_mse:.2f}, MAE: {train_mae:.2f}, R²: {train_r2:.4f}")
print(f"  Validation - MSE: {test_mse:.2f}, MAE: {test_mae:.2f}, R²: {test_r2:.4f}")
print(f"\nAll outputs saved to: {output_dir}")
print("="*80)

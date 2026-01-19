import pandas as pd
import numpy as np
import os
from io import StringIO

# 1. "SCRAPING"
print("Scraping the data...")
md_file_path = "data/data_scraping/dataset_info.md"
with open(md_file_path, 'r') as f:
    lines = f.readlines()

table_lines = [line for line in lines if '|' in line]
table_data = "".join(table_lines)
df_raw = pd.read_csv(StringIO(table_data), sep="|", skipinitialspace=True).dropna(axis=1, how='all')
df_raw.columns = df_raw.columns.str.strip()
df_raw = df_raw.map(lambda x: x.strip() if isinstance(x, str) else x)
print(df_raw.head())

# --- REPORTING DATA: INITIAL COUNT ---
initial_count = len(df_raw)

# 2. CLEANING: DROP NULLS
df_no_nulls = df_raw.dropna()
nulls_dropped = initial_count - len(df_no_nulls)

# 3. NUMERIC CONVERSION & OUTLIER DROPPING
numerical_cols = ['age', 'study_hours', 'class_attendance', 'sleep_hours', 'exam_score']
for col in numerical_cols:
    df_no_nulls[col] = pd.to_numeric(df_no_nulls[col], errors='coerce')

# Drop outliers using Z-score (Threshold = 3)
z_scores = np.abs((df_no_nulls[numerical_cols] - df_no_nulls[numerical_cols].mean()) / df_no_nulls[numerical_cols].std())
df_clean = df_no_nulls[(z_scores < 3).all(axis=1)].copy()
outliers_dropped = len(df_no_nulls) - len(df_clean)

# 4. ENCODING & NORMALIZATION (as discussed before)
df_clean['exam_difficulty'] = df_clean['exam_difficulty'].map({'easy': 0, 'moderate': 1, 'hard': 2})
df_clean['sleep_quality'] = df_clean['sleep_quality'].map({'poor': 0, 'average': 1, 'good': 2})
df_clean['facility_rating'] = df_clean['facility_rating'].map({'low': 0, 'medium': 1, 'high': 2})
df_clean['internet_access'] = df_clean['internet_access'].map({'no': 0, 'yes': 1})
df_final = pd.get_dummies(df_clean, columns=['gender', 'course', 'study_method'], dtype=int)

# 5. SPLITTING AND SCALING
train_df = df_final.sample(frac=0.8, random_state=42)
test_df = df_final.drop(train_df.index)

cols_to_normalize = [col for col in df_final.columns if col not in ['student_id', 'exam_score']]
train_min, train_max = train_df[cols_to_normalize].min(), train_df[cols_to_normalize].max()

train_df[cols_to_normalize] = (train_df[cols_to_normalize] - train_min) / (train_max - train_min)
test_df[cols_to_normalize] = (test_df[cols_to_normalize] - train_min) / (train_max - train_min)

# 6. OUTPUT REPORT
print("--- DATA PREPARATION REPORT ---")
print(f"Total Rows Scraped: {initial_count}")
print(f"Null Values Removed: {nulls_dropped}")
print(f"Outliers Removed (Z-Score > 3): {outliers_dropped}")
print(f"Final Cleaned Dataset Size: {len(df_clean)}")
print(f"Training set size (80%): {len(train_df)}")
print(f"Testing set size (20%): {len(test_df)}")

# Save Files
train_df.to_csv('training_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)
test_df.iloc[[0]].to_csv('activation_data.csv', index=False)
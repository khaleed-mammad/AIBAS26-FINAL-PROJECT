import pandas as pd
import numpy as np
import os
from io import StringIO

# 1. "SCRAPING"
print("Scraping the data...")
print("Reading dataset_info.md to extract tables...")
md_file_path = "data/data_scraping/dataset_info.md"
with open(md_file_path, 'r') as f:
    lines = f.readlines()

table_lines = [line for line in lines if '|' in line]
table_data = "".join(table_lines)
df_raw = pd.read_csv(StringIO(table_data), sep="|", skipinitialspace=True).dropna(axis=1, how='all')
df_raw.columns = df_raw.columns.str.strip()
df_raw = df_raw.map(lambda x: x.strip() if isinstance(x, str) else x)

print("Data scraping completed.")
print(f"Initial dataset shape: {df_raw.shape}")
print(df_raw.head())
# 2. DATA CLEANING & PREPARATION
print("Starting data cleaning and preparation...")
#Check NaNvalues in the data
print("Checking for null values in each column:")
print(df_raw.isnull().sum())

#Since there is no missing data in the dataset, we add a safety check
#  to drop any potential nulls
df_clean = df_raw.dropna()
print(f"Dataset shape after dropping nulls: {df_clean.shape}")

#Print the type of the columns to verify numeric conversion
print("Data types of each column:")
print(df_clean.dtypes)

# Convert numerical columns to appropriate data types
numerical_cols = ['age', 'study_hours', 'class_attendance', 'sleep_hours', 'exam_score']
for col in numerical_cols:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
print("Data types after conversion:")
print(df_clean.dtypes)
# Drop any rows with NaN values after conversion
df_clean = df_clean.dropna()
print(f"Dataset shape after numeric conversion and dropping NaNs: {df_clean.shape}")
# Encode categorical variables- one-hot encoding for nominal, ordinal mapping for ordinal
df_clean['exam_difficulty'] = df_clean['exam_difficulty'].map({'easy': 0, 'moderate': 1, 'hard': 2})
df_clean['sleep_quality'] = df_clean['sleep_quality'].map({'poor': 0, 'average': 1, 'good': 2})
df_clean['facility_rating'] = df_clean['facility_rating'].map({'low': 0, 'medium': 1, 'high': 2})
df_clean['internet_access'] = df_clean['internet_access'].map({'no': 0, 'yes': 1})
df_clean = pd.get_dummies(df_clean, columns=['gender', 'course', 'study_method'], dtype=int)
print("Dataset shape after encoding categorical variables:", df_clean.shape)   

# Drop comlumn student_id as it is not needed for modeling
df_clean = df_clean.drop(columns=['student_id'])

# Descriptive statistics before normalization
print("Descriptive statistics before normalization:")
print(df_clean[numerical_cols].describe())

# Boxplot for numerical columns before normalization
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
df_clean[numerical_cols].boxplot()
plt.title("Boxplot of Numerical Columns Before Normalization")
plt.savefig("boxplot_before_normalization.png")
plt.close()

#We can see that there are no significant outliers in the data from the boxplot. HOwever,
# we will still apply normalization to scale the data for better model performance.

# Normalize numerical columns using z-score scaling except the target column 'exam_score'
cols_to_normalize = [col for col in df_clean.columns if col not in ['exam_score']]
df_clean[cols_to_normalize] = (df_clean[cols_to_normalize] - df_clean[cols_to_normalize].mean()) / df_clean[cols_to_normalize].std()
print("Descriptive statistics after normalization:")
print(df_clean[numerical_cols].describe())

# Save cleaned data
output_dir = "data/data_cleaning"
os.makedirs(output_dir, exist_ok=True)
df_clean.to_csv(os.path.join(output_dir, "joint_data_collection.csv"), index=False)
print("Data cleaning and preparation completed.")

# split into training and testing sets and save
train_df = df_clean.sample(frac=0.8, random_state=42)
test_df = df_clean.drop(train_df.index)
train_df.to_csv(os.path.join(output_dir, "training_data.csv"), index=False)
test_df.to_csv(os.path.join(output_dir, "test_data.csv"), index=False)

# select one test sample and create a activation_data.csv
test_sample = test_df.iloc[[0]]
test_sample.to_csv(os.path.join(output_dir, "activation_data.csv"), index=False)

print("Training, testing, and activation data files created.")
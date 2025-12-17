import kagglehub
import shutil
import os
import pandas as pd

# Download latest version
dataset_name = "kundanbedmutha/exam-score-prediction-dataset"
kaggle_url = f"https://www.kaggle.com/datasets/{dataset_name}"
path = kagglehub.dataset_download(dataset_name)

print("Path to dataset files:", path)

# Copy files to current directory
current_dir = os.getcwd()
data_folder = os.path.join(current_dir, "exam_score_data")

# Create the data folder if it doesn't exist
os.makedirs(data_folder, exist_ok=True)

# Copy all files from the downloaded path to the current directory
csv_files = []
for item in os.listdir(path):
    source = os.path.join(path, item)
    destination = os.path.join(data_folder, item)
    if os.path.isfile(source):
        shutil.copy2(source, destination)
        print(f"Copied {item} to {data_folder}")
        if item.endswith('.csv'):
            csv_files.append(item)

# Create a .md file with dataset information and tables
md_file_path = os.path.join(data_folder, "dataset_info.md")
with open(md_file_path, 'w', encoding='utf-8') as f:
    f.write(f"# Dataset Information\n\n")
    f.write(f"This dataset contains data from the following Kaggle page:\n\n")
    f.write(f"**Dataset:** {dataset_name}\n\n")
    f.write(f"**URL:** [{kaggle_url}]({kaggle_url})\n\n")
    f.write(f"---\n\n")
    
    # Add tables from CSV files
    for csv_file in csv_files:
        f.write(f"## {csv_file}\n\n")
        csv_path = os.path.join(data_folder, csv_file)
        df = pd.read_csv(csv_path)
        f.write(df.head(2500).to_markdown(index=False))
        f.write(f"\n \n")

print(f"\nDataset saved to: {data_folder}")
print(f"Dataset info with tables saved to: {md_file_path}")
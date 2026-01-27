# AIBAS 2025-26 Final Project

This repository is part of the course  
**M. Grum: Advanced AI-based Application Systems**  
Junior Chair for Business Information Science, esp. AI-based Application Systems
University of Potsdam.

## Overview
Exam Score Prediction System using ANN and OLS models deployed in Docker containers. The system compares two regression approaches for predicting student exam scores based on study habits, demographics, and environmental factors.

## Workflow
1. **Data Scraping**: Downloaded dataset from Kaggle using kagglehub API
2. **Data Cleaning**: Z-score normalization, categorical encoding (ordinal + one-hot), train/test split
3. **Model Training**: 
   - **ANN**: 4-layer neural network (128→96→64→32 neurons) with batch normalization and dropout
     - Training: Adam optimizer, early stopping after 64 epochs
     - Performance: Validation MAE = 8.48
   - **OLS**: Linear regression with 24 parameters
     - Performance: Test MAE = 10.86, R² = 0.759
4. **Docker Deployment**: 3-container architecture for model inference

## Docker Architecture
The system uses a shared-volume approach where three containers communicate via a common filesystem:

- **knowledgeBase**: Copies trained models (.keras, .pkl) to `/tmp/knowledgeBase/`
- **activationBase**: Copies input data (CSV) to `/tmp/activationBase/`
- **codeBase**: Reads from `/tmp/`, executes predictions, prints results

**How it works:**
1. All three containers mount the same volume (`ai_system`) at `/tmp/`
2. Knowledge and activation containers populate the volume with their files
3. Code container waits for dependencies, then loads models/data and runs inference
4. Results are output to console, then containers stop

## Quick Start

1. Create the shared volume:
```bash
docker volume create ai_system
```

2. Run OLS prediction:
```bash
cd scenarios/apply_ols_solution
docker-compose up --abort-on-container-exit code
```

3. Run ANN prediction:
```bash
cd scenarios/apply_ann_solution
docker-compose up --abort-on-container-exit code
```

## Project Structure
```
AIBAS26-FINAL-PROJECT/
├── code/
│   ├── kaggle_data_prescrapping.py    - Download data from Kaggle
│   ├── data_cleaning.py               - Preprocess and split data
│   ├── model_ann.py                   - Train neural network
│   ├── model_ols.py                   - Train linear regression
│   ├── apply_ann.py                   - Run ANN inference
│   └── apply_ols.py                   - Run OLS inference
├── data/
│   ├── data_scraping/                 - Raw downloaded data
│   └── data_cleaning/                 - Processed train/test sets
├── images/
│   ├── activationBase_ExamScorePrediction/
│   │   ├── activation_data.csv        - Input data for predictions
│   │   └── Dockerfile
│   ├── codeBase_ExamScorePrediction/
│   │   ├── apply_ann.py               - Inference scripts for predictions
│   │   ├── apply_ols.py
│   │   └── Dockerfile
│   ├── knowledgeBase_ExamScorePrediction/
│   │   ├── currentAiSolution.keras    - Trained ANN model
│   │   ├── currentOlsSolution.pkl     - Trained OLS model
│   │   └── Dockerfile
│   └── learningBase_ExamScorePrediction/
│       ├── training_data.csv          - Training/test data
│       ├── test_data.csv
│       └── Dockerfile
├── scenarios/
│   ├── apply_ols_solution/
│   │   └── docker-compose.yml         - OLS deployment config
│   └── apply_ann_solution/
│       └── docker-compose.yml         - ANN deployment config
├── documentation/
│   ├── ann_training_report.txt        - Results of running model_ann.py
│   └── ols_training_report.txt        - Results of running model_ols.py
└── visualizations/                    - Plots of data and model performance
    ├── ann_diagnostic_plots.pdf
    ├── ann_scatter_plots.pdf
    ├── ols_diagnostic_plots.png
    └── ols_scatter_plots.png
```

## Authors
- Irene Gema Castillo Mansilla
- Khalid Mammadov

## License
AGPL-3.0

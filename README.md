# AIBAS 2025-26 Final Project

This repository is part of the course  
**M. Grum: Advanced AI-based Application Systems**  
Junior Chair for Business Information Science, esp. AI-based Application Systems
University of Potsdam.

## Overview
Exam Score Prediction System using machine learning models deployed in Docker containers. The system compares ANN and OLS regression models for predicting student exam scores based on study habits, demographics, and environmental factors.

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

2. Run predictions:
```bash
cd scenarios/apply_ols_solution
docker-compose up --abort-on-container-exit code
```

## Project Structure
```
code/         - Python scripts (scraping, cleaning, training, inference)
data/         - Raw and processed datasets
images/       - Dockerfiles for container images
scenarios/    - Docker Compose configurations
```

## Authors
- Irene Gema Castillo Mansilla
- Khalid Mammadov

## License
AGPL-3.0

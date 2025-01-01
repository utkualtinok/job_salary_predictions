# Salary Prediction Project

This repository contains a machine learning project for predicting salaries based on various features using multiple regression models. The project also includes a Streamlit web application for interactive salary prediction.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Streamlit App](#running-the-streamlit-app)
- [Project Structure](#project-structure)
- [Models Used](#models-used)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Overview

This project focuses on salary prediction using a dataset of job roles in the data science domain. The project demonstrates the end-to-end process, from data preprocessing and feature engineering to model training, hyperparameter tuning, and deployment via a Streamlit web application.

---

## Features

- **Data Loading and Preprocessing**: Handles missing values, encodes categorical variables, and splits the data into features and target.
- **Model Evaluation**: Compares the performance of multiple regression models.
- **Hyperparameter Tuning**: Optimizes model parameters using GridSearchCV.
- **Feature Importance Plotting**: Visualizes the most important features for the prediction task.
- **Interactive Web App**: Streamlit-based app for predicting salaries based on user inputs.

---

## Requirements

- Python 3.7+
- Libraries:
  - numpy
  - pandas
  - scikit-learn
  - catboost
  - lightgbm
  - matplotlib
  - seaborn
  - joblib
  - streamlit

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/salary-prediction.git
   cd salary-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Running the Streamlit App

1. Ensure the dataset (`ds_salaries.csv`) is placed in the `Datasets` directory of the project repository:
   ```
   ./Datasets/ds_salaries.csv
   ```

2. Train the model (if not already trained):
   ```bash
   python train_model.py
   ```

3. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. Access the app in your browser at:
   ```
   http://localhost:8501
   ```

---

## Project Structure

```
.
├── app.py                 # Streamlit web app
├── train_model.py         # Training script
├── requirements.txt       # Required libraries
├── Datasets/              # Directory for datasets
│   └── ds_salaries.csv    # Dataset
├── salary_prediction_model.joblib  # Trained model
└── README.md              # Project documentation
```

---

## Models Used

The following machine learning models were evaluated:

- **Linear Regression**
- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **CatBoost Regressor**
- **LightGBM Regressor**

### Best Performing Model

- **LightGBM Regressor** with hyperparameter tuning achieved the best results in terms of RMSE.

---

## Results

- RMSE of best model: **[Insert Final RMSE Score]**
- Feature importance visualization highlights the most influential predictors.

---

## Future Improvements

- Implement additional preprocessing techniques (e.g., scaling and feature selection).
- Add support for other machine learning models.
- Enhance the Streamlit UI for better user experience.
- Integrate real-time dataset updates.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

# Imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# Function to load and inspect the dataset
def load_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
    print(f"Columns: {df.columns.tolist()}")
    return df


# Function to preprocess data
def preprocess_data(df, drop_columns, target_col):
    # Drop specified columns
    df = df.drop(columns=drop_columns, errors='ignore')

    # Encode categorical variables
    df = pd.get_dummies(df, drop_first=True)

    # Separate features and target variable
    y = df[target_col]
    X = df.drop(target_col, axis=1)
    return X, y


# Function to evaluate models
def evaluate_models(models, X, y, scoring="neg_mean_squared_error"):
    results = {}
    for name, model in models:
        scores = -cross_val_score(model, X, y, cv=5, scoring=scoring, n_jobs=-1)
        results[name] = np.mean(np.sqrt(scores))  # RMSE
    return results


# Function to plot feature importance
def plot_feature_importance(model, X, num=10):
    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": X.columns})
    feature_imp = feature_imp.sort_values(by="Value", ascending=False)[:num]
    plt.figure(figsize=(8, 6))
    sns.barplot(x="Value", y="Feature", data=feature_imp)
    plt.title("Top Features")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()


# Hyperparameter tuning function
def tune_model(model, param_grid, X, y):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1, verbose=True)
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    print(f"Best Parameters: {grid_search.best_params_}")
    return best_model




# Load dataset
df = load_data("C:/Users/utku/Desktop/final_projects/Datasets/ds_salaries.csv")

# Preprocessing
drop_columns = ["salary", "job_title", "salary_currency", "employee_residence", "company_location"]
X, y = preprocess_data(df, drop_columns, target_col="salary_in_usd")

# Define models
models = [
    ("LinearRegression", LinearRegression()),
    ("RandomForest", RandomForestRegressor(random_state=42)),
    ("GradientBoosting", GradientBoostingRegressor(random_state=42)),
    ("CatBoost", CatBoostRegressor(verbose=False, random_state=42)),
    ("LightGBM", LGBMRegressor(random_state=42))
]

# Evaluate models
results = evaluate_models(models, X, y)
print("Model Results:")
for name, score in results.items():
    print(f"{name}: RMSE = {round(score, 2)}")



# Hyperparameter tuning for CatBoost
catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}
final_catboost = tune_model(CatBoostRegressor(verbose=False, random_state=42), catboost_params, X, y)

print("Final CatBoost Parameters:", final_catboost.get_params())


# Hyperparameter tuning for LightGBM
lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}
final_lgbm = tune_model(LGBMRegressor(random_state=42), lgbm_params, X, y)

print("Final LightGBM Parameters:", final_lgbm.get_params())




# Train the best model (using CatBoost as an example here)
final_model = final_lgbm
final_model.fit(X, y)


rmse_scores = -cross_val_score(final_lgbm, X, y, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
rmse = np.sqrt(rmse_scores.mean())

print(f"Final LightGBM RMSE (after tuning): {rmse:.2f}")


# Plot feature importance
plot_feature_importance(final_model, X)

# Save the model
model_filename = "salary_prediction_model.joblib"
joblib.dump(final_model, model_filename)
print(f"Model saved as {model_filename}")

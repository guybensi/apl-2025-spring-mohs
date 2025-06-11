
import pandas as pd
import numpy as np
import optuna
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import median_absolute_error

# === Load data ===
df = pd.read_csv("train.csv")
df = df[df["Hardness"] <= 10]  # Remove outliers

X = df.drop(columns=["id", "Hardness"])
y = df["Hardness"]

# Impute missing values
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === Objective function for Optuna ===
def objective(trial):
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "random_state": 42
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return median_absolute_error(y_val, preds)

# === Run tuning ===
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

print("✅ Best parameters:")
print(study.best_params)
print("✅ Best median absolute error:", study.best_value)

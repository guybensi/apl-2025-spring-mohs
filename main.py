import pandas as pd
import optuna
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor

# Load data
train_df = pd.read_csv("train.csv")


X = train_df.drop(columns=["id", "Hardness"])
y = train_df["Hardness"]

# Optuna objective functions
def objective_histgb(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "max_iter": trial.suggest_int("max_iter", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
        "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 5.0)
    }
    model = HistGradientBoostingRegressor(random_state=42, **params)
    score = cross_val_score(model, X, y, scoring="neg_median_absolute_error", cv=KFold(n_splits=5)).mean()
    return -score

def objective_gb(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 15)
    }
    model = GradientBoostingRegressor(random_state=42, **params)
    score = cross_val_score(model, X, y, scoring="neg_median_absolute_error", cv=KFold(n_splits=5)).mean()
    return -score

def objective_xgb(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0)
    }
    model = XGBRegressor(random_state=42, verbosity=0, **params)
    score = cross_val_score(model, X, y, scoring="neg_median_absolute_error", cv=KFold(n_splits=5)).mean()
    return -score

# Optimize all models
study_histgb = optuna.create_study(direction="minimize")
study_histgb.optimize(objective_histgb, n_trials=30)

study_gb = optuna.create_study(direction="minimize")
study_gb.optimize(objective_gb, n_trials=30)

study_xgb = optuna.create_study(direction="minimize")
study_xgb.optimize(objective_xgb, n_trials=30)

# Results
print("Best MedAE:")
print("HistGradientBoosting:", study_histgb.best_value)
print("GradientBoosting:", study_gb.best_value)
print("XGBoost:", study_xgb.best_value)

print("\nBest Parameters:")
print("HistGB:", study_histgb.best_params)
print("GB:", study_gb.best_params)
print("XGB:", study_xgb.best_params)

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
import xgboost as xgb

# === טוען את הקבצים ===
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# סינון לפי hardness <= 10
filtered_train_df = train_df[train_df["Hardness"] <= 10]
X_train = filtered_train_df.drop(columns=["id", "Hardness"])
y_train = filtered_train_df["Hardness"]
X_test = test_df.drop(columns=["id"])

# === פרמטרים אופטימליים לכל מודל ===
best_params = {
    "gbr": {
        "learning_rate": 0.02654,
        "n_estimators": 532,
        "max_depth": 6
    },
    "xgb": {
        "learning_rate": 0.1359,
        "n_estimators": 646,
        "max_depth": 4,
        "subsample": 0.7927,
        "colsample_bytree": 0.5956,
        "gamma": 0.0605
    },
    "hgb": {
        "learning_rate": 0.2299,
        "max_iter": 200,
        "max_depth": 13,
        "min_samples_leaf": 39,
        "l2_regularization": 0.6818
    }
}

# === אימון מודלים ===
model_gbr = GradientBoostingRegressor(**best_params["gbr"], random_state=42)
model_xgb = xgb.XGBRegressor(**best_params["xgb"], random_state=42)
model_hgb = HistGradientBoostingRegressor(**best_params["hgb"], random_state=42)

model_gbr.fit(X_train, y_train)
model_xgb.fit(X_train, y_train)
model_hgb.fit(X_train, y_train)

# === תחזיות ===
preds_gbr = model_gbr.predict(X_test)
preds_xgb = model_xgb.predict(X_test)
preds_hgb = model_hgb.predict(X_test)

# === ממוצע משוקלל ===
w_gbr, w_xgb, w_hgb = 0.5, 0.3, 0.2
final_preds = w_gbr * preds_gbr + w_xgb * preds_xgb + w_hgb * preds_hgb

# === הגשת התחזיות ===
submission = pd.DataFrame({
    "id": test_df["id"],
    "Hardness": final_preds
})
submission.to_csv("final_submission_filtered.csv", index=False)
print("✅ Saved to final_submission_filtered.csv")

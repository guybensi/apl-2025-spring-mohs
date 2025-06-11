
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb

# === Load data ===
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Remove outliers
train_df = train_df[train_df["Hardness"] <= 10]

X_raw = train_df.drop(columns=["id", "Hardness"])
y = train_df["Hardness"]
X_test_raw = test_df.drop(columns=["id"])

# Impute missing values and restore DataFrame structure
imputer = SimpleImputer(strategy="mean")
X = pd.DataFrame(imputer.fit_transform(X_raw), columns=X_raw.columns)
X_test = pd.DataFrame(imputer.transform(X_test_raw), columns=X_test_raw.columns)

# Discretize target
bin_edges = np.linspace(0, 10, 21)
bin_labels = np.arange(len(bin_edges) - 1)
y_binned = pd.cut(y, bins=bin_edges, labels=bin_labels).astype(int)

# Train/val split
X_train, X_val, y_train_binned, y_val_binned, y_train_cont, y_val_cont = train_test_split(
    X, y_binned, y, test_size=0.2, random_state=42)

# === LightGBM classification model with tuned parameters ===
clf = lgb.LGBMClassifier(
    num_leaves=50,
    max_depth=12,
    learning_rate=0.0187,
    n_estimators=833,
    min_child_samples=46,
    subsample=0.5162,
    colsample_bytree=0.9079,
    random_state=42
)
clf.fit(X_train, y_train_binned)

# Predict class probabilities (test set)
bin_midpoints = 0.5 * (bin_edges[:-1] + bin_edges[1:])
num_bins = len(bin_midpoints)

class_probs_raw = clf.predict_proba(X_test)
proba = np.zeros((class_probs_raw.shape[0], num_bins))
for idx, c in enumerate(clf.classes_):
    proba[:, c] = class_probs_raw[:, idx]
reg_preds_class = proba.dot(bin_midpoints)

# GradientBoosting regressor
gbr = GradientBoostingRegressor(learning_rate=0.03, n_estimators=400, max_depth=6, random_state=42)
gbr.fit(X_train, y_train_cont)
reg_preds_gbr = gbr.predict(X_test)

# Meta-model training on validation set
val_probs_raw = clf.predict_proba(X_val)
proba_val = np.zeros((val_probs_raw.shape[0], num_bins))
for idx, c in enumerate(clf.classes_):
    proba_val[:, c] = val_probs_raw[:, idx]
val_preds_class = proba_val.dot(bin_midpoints)
val_preds_gbr = gbr.predict(X_val)

meta_X_val = np.vstack([val_preds_class, val_preds_gbr]).T
meta_model = Ridge(alpha=1.0)
meta_model.fit(meta_X_val, y_val_cont)

# Final prediction
stacked_test = np.vstack([reg_preds_class, reg_preds_gbr]).T
final_preds = meta_model.predict(stacked_test)
final_preds = np.clip(final_preds, 0, 10)

# Output
submission = pd.DataFrame({
    "id": test_df["id"],
    "Hardness": final_preds
})
submission.to_csv("final_submission_ensemble.csv", index=False)
print("✅ Saved to final_submission_ensemble.csv")

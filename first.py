import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import median_absolute_error

# Load data
train_df = pd.read_csv("train.csv")

# 🔍 Filter out extreme values (Hardness > 8)
train_df = train_df[train_df["Hardness"] <= 10].copy()

# Prepare features and target
X = train_df.drop(columns=["id", "Hardness"])
y = train_df["Hardness"]

# Holdout split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model with best Optuna params
model = HistGradientBoostingRegressor(
    learning_rate=0.14336,
    max_iter=393,
    max_depth=7,
    min_samples_leaf=43,
    l2_regularization=0.8360,
    random_state=42
)

# Holdout evaluation
model.fit(X_train, y_train)
val_preds = model.predict(X_val)
holdout_mae = median_absolute_error(y_val, val_preds)
print(f"📏 Holdout MedAE: {holdout_mae:.5f}")

# 5-fold CV evaluation
cv_scores = -cross_val_score(model, X, y, cv=5, scoring="neg_median_absolute_error")
cv_mae = cv_scores.mean()
print(f"📏 Cross-Validation MedAE (5-fold): {cv_mae:.5f}")
print(f"Individual folds: {cv_scores}")

# Plot filtered Hardness distribution
y.hist(bins=30)
plt.title("Hardness Distribution (≤ 10 only)")
plt.xlabel("Hardness")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("hardness_distribution_filtered.png")
print("📊 Saved histogram as hardness_distribution_filtered.png")

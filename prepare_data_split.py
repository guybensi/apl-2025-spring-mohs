import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Load train.csv
df = pd.read_csv("train.csv")
X = df.drop(columns=["id", "Hardness"])
y = df["Hardness"]

# Fill missing values
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

# Split to train/val
X_train, X_val, y_train, y_val = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Save to disk
np.save("X_train.npy", X_train)
np.save("X_val.npy", X_val)
np.save("y_train.npy", y_train.to_numpy())
np.save("y_val.npy", y_val.to_numpy())

print("✅ Data prepared and saved as .npy files")

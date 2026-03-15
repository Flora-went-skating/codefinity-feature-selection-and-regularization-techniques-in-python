from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.metrics import r2_score

# 1. Load dataset
X, y = load_diabetes(return_X_y=True)

# 2. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Define Ridge pipeline
ridge_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", Ridge(alpha=1.0))
])

# 4. Define Lasso pipeline
lasso_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", Lasso(alpha=0.01, random_state=42))
])

# 5. Fit and evaluate both models
ridge_pipe.fit(X_train, y_train)
lasso_pipe.fit(X_train, y_train)

ridge_pred = ridge_pipe.predict(X_test)
lasso_pred = lasso_pipe.predict(X_test)

ridge_r2 = r2_score(y_test, ridge_pred)
lasso_r2 = r2_score(y_test, lasso_pred)

ridge_norm = np.linalg.norm(ridge_pipe.named_steps["regressor"].coef_, ord=2)
lasso_norm = np.linalg.norm(lasso_pipe.named_steps["regressor"].coef_, ord=1)

print(f"Ridge R²: {ridge_r2:.3f}")
print(f"Lasso R²: {lasso_r2:.3f}")
print(f"Ridge coef norm: {ridge_norm:.2f}")
print(f"Lasso coef norm: {lasso_norm:.2f}")
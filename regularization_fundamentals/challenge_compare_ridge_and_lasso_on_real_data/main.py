from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# 1. Load dataset
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Initialize models
ridge = Ridge(alpha=1.0, random_state=42)
lasso = Lasso(alpha=0.1, random_state=42)

# 3. Fit models
ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)

# 4. Make predictions
ridge_preds = ridge.predict(X_test)
lasso_preds = lasso.predict(X_test)

# 5. Evaluate models
ridge_r2 = r2_score(y_test, ridge_preds)
ridge_mse = mean_squared_error(y_test, ridge_preds)
lasso_r2 = r2_score(y_test, lasso_preds)
lasso_mse = mean_squared_error(y_test, lasso_preds)

# 6. Create DataFrame for coefficients
coefs = pd.DataFrame({
    'Feature': range(X.shape[1]),
    'Ridge': ridge.coef_,
    'Lasso': lasso.coef_
})

# 7. Display results
print("Ridge -> R²:", ridge_r2, "MSE:", ridge_mse)
print("Lasso -> R²:", lasso_r2, "MSE:", lasso_mse)
print("\nCoefficients Comparison:")
print(coefs)
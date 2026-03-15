from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

# 1. Load dataset
X, y = load_diabetes(return_X_y=True)

# 2. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

# 3. Define Lasso-based feature selector
feature_selector = SelectFromModel(Lasso(alpha=0.01, random_state=42))

# 4. Build pipeline: scaling -> feature selection -> regression
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("selector", feature_selector),
    ("regressor", LinearRegression())
])

# 5. Train and evaluate
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred)
selected_features = pipeline.named_steps["selector"].get_support().sum()

print(f"R² score: {r2:.3f}")
print(f"Selected features: {selected_features}")
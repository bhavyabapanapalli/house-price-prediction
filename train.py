
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# STEP 1: Load dataset
data = pd.read_csv("data/Housing.csv")

print("Dataset loaded:", data.shape)

# STEP 2: Separate input features and target
X = data.drop(columns=["price"])
y = data["price"]

# STEP 3: Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

# STEP 4: Preprocess the data
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# STEP 5: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# STEP 6: Define models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
}

results = {}
trained_models = {}

# STEP 7: Train and evaluate each model
for name, estimator in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", estimator)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results[name] = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }
    trained_models[name] = pipeline

    print(f"\n{name}")
    print(f"MAE:  ₹{mae:,.0f}")
    print(f"RMSE: ₹{rmse:,.0f}")
    print(f"R²:   {r2:.4f}")

# STEP 8: Select the model with the lowest test MAE
best_name = min(results, key=lambda name: results[name]["MAE"])
best_model = trained_models[best_name]

print("\nMODEL COMPARISON")
print(pd.DataFrame(results).T.round(4))
print("\nBest model by test MAE:", best_name)

# STEP 9: Save the winning model
import os
os.makedirs("models", exist_ok=True)

joblib.dump(best_model, "models/house_price_model.pkl")

print("\nSaved model to models/house_price_model.pkl")

# STEP 10: Test a sample prediction
sample_house = X_test.iloc[[0]]
actual_price = y_test.iloc[0]
predicted_price = best_model.predict(sample_house)[0]

print("\nSAMPLE PREDICTION")
print("Actual price:    ₹{:,.0f}".format(actual_price))
print("Predicted price: ₹{:,.0f}".format(predicted_price))


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# STEP 1: Load the dataset
data = pd.read_csv("data/House Price Prediction Dataset.csv")

print("\nFIRST 5 ROWS")
print(data.head())

# STEP 2: Explore the dataset
print("\nDATASET INFORMATION")
data.info()

print("\nMISSING VALUES")
print(data.isnull().sum())

print("\nDUPLICATE ROWS")
print(data.duplicated().sum())

print("\nSTATISTICAL SUMMARY")
print(data.describe())

# STEP 3: Visualize Area vs Price
plt.figure(figsize=(8, 5))
plt.scatter(data["Area"], data["Price"], alpha=0.6)
plt.xlabel("Area (square feet)")
plt.ylabel("House Price")
plt.title("House Area vs Price")
plt.tight_layout()
plt.show()

# STEP 4: Visualize Area vs Price by Location
plt.figure(figsize=(8, 5))

for location in data["Location"].unique():
    subset = data[data["Location"] == location]

    plt.scatter(
        subset["Area"],
        subset["Price"],
        label=location,
        alpha=0.6
    )

plt.xlabel("Area (square feet)")
plt.ylabel("House Price")
plt.title("House Area vs Price by Location")
plt.legend()
plt.tight_layout()
plt.show()

# STEP 5: Separate input features and target

X = data.drop(columns=["Id", "Price"])
y = data["Price"]

print("\nORIGINAL INPUT FEATURES")
print(X.head())

print("\nTARGET VARIABLE")
print(y.head())

# STEP 6: Convert categorical columns into numerical columns
categorical_columns = ["Location", "Condition", "Garage"]

X = pd.get_dummies(
    X,
    columns=categorical_columns,
    dtype=int
)

print("\nENCODED FEATURES")
print(X.head())
print("\nTotal input features:", X.shape[1])

# STEP 7: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTRAIN-TEST SPLIT")
print("Training features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)
print("Training target shape:", y_train.shape)
print("Testing target shape:", y_test.shape)

print("\nData preparation completed successfully!")


# STEP 8: Train a Linear Regression model

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

print("\n========== MODEL TRAINING COMPLETE ==========")

# STEP 9: Predict house prices on test data
predictions = model.predict(X_test)

print("\nFIRST 5 PREDICTIONS")
for actual, predicted in zip(y_test.iloc[:5], predictions[:5]):
    print(f"Actual Price: {actual} | Predicted Price: {predicted:.2f}")

# STEP 10: Evaluate the model
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("\nMODEL EVALUATION")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")


# STEP 11: Compare with Random Forest

from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
rf_r2 = r2_score(y_test, rf_predictions)

print("\n========== RANDOM FOREST RESULTS ==========")
print(f"MAE:  {rf_mae:.2f}")
print(f"RMSE: {rf_rmse:.2f}")
print(f"R² Score: {rf_r2:.4f}")

print("\n========== MODEL COMPARISON ==========")
print(f"Linear Regression MAE: {mae:.2f}")
print(f"Random Forest MAE:     {rf_mae:.2f}")



# STEP 12: Investigate relationships with house price

print("\n========== NUMERICAL FEATURE CORRELATIONS ==========")

numeric_data = data[
    ["Area", "Bedrooms", "Bathrooms", "Floors", "YearBuilt", "Price"]
]

print(
    numeric_data.corr()["Price"]
    .sort_values(ascending=False)
)

print("\n========== AVERAGE PRICE BY LOCATION ==========")
print(data.groupby("Location")["Price"].mean().round(2))

print("\n========== AVERAGE PRICE BY CONDITION ==========")
print(data.groupby("Condition")["Price"].mean().round(2))

print("\n========== BASELINE MODEL ==========")

from sklearn.dummy import DummyRegressor

baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_test)

baseline_mae = mean_absolute_error(y_test, baseline_predictions)
baseline_r2 = r2_score(y_test, baseline_predictions)

print(f"Baseline MAE: {baseline_mae:.2f}")
print(f"Baseline R²:  {baseline_r2:.4f}")


# STEP 13: Check price variability

print("\n========== PRICE VARIABILITY ==========")

print("Price standard deviation:", data["Price"].std())
print("Price variance:", data["Price"].var())
print("Unique prices:", data["Price"].nunique())

print("\n========== BASELINE COMPARISON ==========")

print(f"Baseline MAE: {baseline_mae:.2f}")
print(f"Linear Regression MAE: {mae:.2f}")
print(f"Random Forest MAE: {rf_mae:.2f}")

print("\nPercentage improvement over baseline:")

improvement = (
    (baseline_mae - mae) / baseline_mae
) * 100

print(f"{improvement:.2f}%")

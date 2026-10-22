
import pandas as pd

data = pd.read_csv("data/Housing.csv")

print("DATASET SHAPE:")
print(data.shape)

print("\nFIRST 5 ROWS:")
print(data.head())

print("\nCOLUMN NAMES:")
print(data.columns.tolist())

print("\nDATA TYPES:")
print(data.dtypes)

print("\nMISSING VALUES:")
print(data.isnull().sum())

print("\nDUPLICATE ROWS:")
print(data.duplicated().sum())

print("\nPRICE STATISTICS:")
print(data["price"].describe())

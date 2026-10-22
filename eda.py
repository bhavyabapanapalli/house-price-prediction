
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data/Housing.csv")

# 1. Understand the price distribution
plt.figure(figsize=(8, 5))
plt.hist(data["price"] / 100000, bins=20, edgecolor="black")
plt.xlabel("House Price (Lakh)")
plt.ylabel("Number of Houses")
plt.title("Distribution of House Prices")
plt.tight_layout()
plt.show()

# 2. Area vs Price
plt.figure(figsize=(8, 5))
plt.scatter(data["area"], data["price"] / 100000, alpha=0.6)
plt.xlabel("Area (square feet)")
plt.ylabel("House Price (Lakh)")
plt.title("Area vs House Price")
plt.tight_layout()
plt.show()

# 3. Average price by furnishing status
furnishing_prices = (
    data.groupby("furnishingstatus")["price"].mean() / 100000
)
print("\nAverage price by furnishing status (Lakh):")
print(furnishing_prices)

furnishing_prices.plot(kind="bar", figsize=(8, 5))
plt.ylabel("Average House Price (Lakh)")
plt.title("Furnishing Status vs Average Price")
plt.tight_layout()
plt.show()

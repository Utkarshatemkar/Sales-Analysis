# =========================================================
# 📊 SALES ANALYSIS + ML PROJECT (BEGINNER TO INTERVIEW READY)
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

print("\n🚀 PROJECT STARTED")

# =========================================================
# 📌 1. LOAD DATA
# =========================================================
df = pd.read_csv("data/sales_data.csv")
print("\n✅ Data Loaded")

# =========================================================
# 📌 2. BASIC INFORMATION
# =========================================================
print("\n📊 FIRST 5 ROWS:\n", df.head())
print("\n📊 INFO:")
print(df.info())

print("\n📊 DESCRIPTION:\n", df.describe())

# =========================================================
# 📌 3. DATA CLEANING
# =========================================================
df.columns = df.columns.str.strip()

# Convert date
df["Date"] = pd.to_datetime(df["Date"])

# Remove duplicates
df = df.drop_duplicates()

# Handle missing values
df["Sales"] = df["Sales"].fillna(df["Sales"].mean())
df["Profit"] = df["Profit"].fillna(df["Profit"].mean())
df["Quantity"] = df["Quantity"].fillna(df["Quantity"].median())

print("\n✅ Data Cleaning Completed")

# =========================================================
# 📌 4. FEATURE ENGINEERING
# =========================================================
df["Profit_Margin"] = (df["Profit"] / df["Sales"]) * 100
df["Month"] = df["Date"].dt.month
df["Year"] = df["Date"].dt.year
df["Day_Name"] = df["Date"].dt.day_name()

df["Sales_Category"] = pd.cut(
    df["Sales"],
    bins=[0, 1000, 2000, 3000, 10000],
    labels=["Low", "Medium", "High", "Very High"]
)

print("\n✅ Feature Engineering Done")

# =========================================================
# 📌 5. DATA QUALITY CHECK
# =========================================================
print("\n🔍 DATA QUALITY REPORT")
print("Missing Values:\n", df.isnull().sum())
print("Duplicate Rows:", df.duplicated().sum())
print("Negative Sales:", (df["Sales"] < 0).sum())

# =========================================================
# 📌 6. GROUPED ANALYSIS
# =========================================================
region_sales = df.groupby("Region")["Sales"].sum()
product_sales = df.groupby("Product")["Sales"].sum()
product_profit = df.groupby("Product")["Profit"].sum()
monthly_sales = df.groupby("Month")["Sales"].sum()

print("\n📊 REGION SALES:\n", region_sales)
print("\n📊 PRODUCT SALES:\n", product_sales)

# =========================================================
# 📌 7. VISUALIZATION
# =========================================================

# Sales by Region
plt.figure()
region_sales.plot(kind="bar")
plt.title("Sales by Region")
plt.xlabel("Region")
plt.ylabel("Sales")
plt.tight_layout()
plt.savefig("outputs/region_sales.png")
plt.close()

# Monthly Sales
plt.figure()
monthly_sales.plot(kind="line", marker="o")
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.tight_layout()
plt.savefig("outputs/monthly_sales.png")
plt.close()

# Product Profit
plt.figure()
product_profit.plot(kind="bar", color="green")
plt.title("Profit by Product")
plt.xlabel("Product")
plt.ylabel("Profit")
plt.tight_layout()
plt.savefig("outputs/product_profit.png")
plt.close()

print("\n📊 Visualizations Saved")

# =========================================================
# 📌 8. MACHINE LEARNING MODEL
# =========================================================

X = df[["Sales", "Quantity"]]
y = df["Profit"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

print("\n🤖 MODEL TRAINED")
print("Accuracy:", accuracy)

# Save model
joblib.dump(model, "outputs/profit_model.pkl")

# =========================================================
# 📌 9. PREDICTION
# =========================================================
df["Predicted_Profit"] = model.predict(X)

print("\n🔮 Predictions Added")

# =========================================================
# 📌 10. FUTURE PREDICTION EXAMPLE
# =========================================================
future_data = pd.DataFrame({
    "Sales": [1200, 2500, 4000],
    "Quantity": [3, 5, 8]
})

future_data["Predicted_Profit"] = model.predict(future_data)

future_data.to_csv("outputs/future_predictions.csv", index=False)

print("\n📈 Future Prediction Done")

# =========================================================
# 📌 11. SALES INSIGHTS
# =========================================================

top_region = region_sales.idxmax()
top_product = product_sales.idxmax()

print("\n🏆 TOP REGION:", top_region)
print("🏆 TOP PRODUCT:", top_product)

# =========================================================
# 📌 12. EXPORT FILES
# =========================================================
df.to_csv("outputs/cleaned_data.csv", index=False)
region_sales.to_csv("outputs/region_sales.csv")
product_sales.to_csv("outputs/product_sales.csv")
monthly_sales.to_csv("outputs/monthly_sales.csv")

print("\n💾 All CSV files exported")

# =========================================================
# 📌 13. FINAL SUMMARY
# =========================================================
print("\n==============================")
print("🎉 PROJECT COMPLETED SUCCESSFULLY")
print("==============================")
print("✔ Data Cleaning Done")
print("✔ Feature Engineering Done")
print("✔ Analysis Done")
print("✔ ML Model Trained")
print("✔ Predictions Done")
print("✔ Outputs Saved")

print("\n🚀 READY FOR GITHUB + RESUME")

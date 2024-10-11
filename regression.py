import pandas as pd
import statsmodels.api as sm

# Dataset
data = {
    "Chi phí sạc / điểm sạc": [39.78, 42.14, 43.21, 38.71, 40.29, 38.47, 43.50, 43.11, 37.37, 38.86, 35.47, 38.32, 31.13, 44.20, 33.90, 31.97, 36.58, 34.21, 24.45, 19.69, 25.62, 19.24],
    "Chi phí vận hành / điểm sạc": [32.34, 34.26, 35.12, 31.46, 32.75, 31.27, 35.36, 35.04, 30.38, 31.59, 28.83, 31.15, 25.30, 35.93, 27.56, 25.99, 29.74, 27.81, 19.88, 16.01, 20.82, 15.64],
    "Chi phí đi lại / điểm sạc": [10.93, 10.58, 9.64, 18.54, 15.61, 20.08, 10.57, 12.20, 24.79, 22.33, 28.79, 23.30, 37.53, 13.76, 32.97, 36.89, 28.51, 33.39, 52.03, 60.95, 50.25, 62.10],
    "Chi phí chờ / điểm sạc": [8.24, 6.37, 5.85, 6.00, 5.24, 5.15, 4.53, 4.33, 3.15, 3.04, 3.23, 2.46, 2.68, 1.78, 2.34, 2.05, 1.71, 1.46, 1.33, 1.48, 0.97, 1.24],
    "Chi phí lắp đặt / điểm sạc": [4.13, 4.05, 4.06, 3.74, 3.76, 3.61, 3.91, 3.85, 3.26, 3.36, 3.12, 3.24, 2.72, 3.59, 2.89, 2.70, 3.00, 2.79, 2.03, 1.69, 2.07, 1.63],
    "Chi phí thuê đất / điểm sạc": [4.06, 2.15, 1.69, 1.13, 1.98, 1.05, 1.78, 1.12, 0.78, 0.55, 0.28, 1.30, 0.41, 0.51, 0.13, 0.20, 0.26, 0.17, 0.14, 0.04, 0.14, 0.04]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Define the dependent variable and independent variables
X = df.drop(columns=["Chi phí sạc / điểm sạc"])
y = df["Chi phí sạc / điểm sạc"]

# Add a constant to the independent variables matrix (intercept)
X = sm.add_constant(X)

# Fit an Ordinary Least Squares (OLS) regression model
model = sm.OLS(y, X).fit()

# Get the summary of the model
model_summary = model.summary()

model_summary

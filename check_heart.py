import pandas as pd

# Load heart disease dataset
data = pd.read_csv(
    "data/heart.csv",
    header=None
)

print("Dataset loaded successfully!")

print("\nDataset shape:")
print(data.shape)

print("\nFirst 5 rows:")
print(data.head())

print("\nNumber of columns:")
print(len(data.columns))
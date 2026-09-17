import pandas as pd

# ============================================
# LOAD ORIGINAL HEART DATASET
# ============================================

data = pd.read_csv(
    "data/heart.csv",
    header=None
)

print("Original dataset loaded!")
print("Shape:", data.shape)


# ============================================
# ADD COLUMN NAMES
# ============================================

data.columns = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target"
]


# ============================================
# DISPLAY COLUMN NAMES
# ============================================

print("\nColumn names:")
print(data.columns.tolist())


# ============================================
# HANDLE MISSING VALUES
# ============================================

data = data.replace("?", pd.NA)

print("\nMissing values:")
print(data.isnull().sum())


# ============================================
# CONVERT COLUMNS TO NUMERIC
# ============================================

for column in data.columns:

    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )


# ============================================
# FILL MISSING VALUES
# ============================================

for column in data.columns:

    if data[column].isnull().sum() > 0:

        data[column] = data[column].fillna(
            data[column].median()
        )


# ============================================
# CONVERT TARGET TO BINARY
# ============================================

data["target"] = (
    data["target"] > 0
).astype(int)


# ============================================
# SHOW TARGET DISTRIBUTION
# ============================================

print("\nTarget distribution:")
print(data["target"].value_counts())


# ============================================
# SAVE CLEAN DATASET
# ============================================

data.to_csv(
    "data/heart_clean.csv",
    index=False
)

print("\nClean heart dataset saved successfully!")

print("\nFinal dataset shape:")
print(data.shape)

print("\nFirst 5 rows:")
print(data.head())
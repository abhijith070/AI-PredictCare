import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# LOAD DATASET
# ============================================

data = pd.read_csv("data/diabetes.csv")

print("Dataset loaded successfully!")
print("Shape:", data.shape)

# ============================================
# 1. OUTCOME DISTRIBUTION
# ============================================

plt.figure(figsize=(6, 4))

sns.countplot(
    x="Outcome",
    data=data
)

plt.title("Diabetes Outcome Distribution")
plt.xlabel("Outcome (0 = No Diabetes, 1 = Diabetes)")
plt.ylabel("Number of Patients")

plt.show()

# ============================================
# 2. GLUCOSE DISTRIBUTION
# ============================================

plt.figure(figsize=(7, 4))

sns.histplot(
    data["Glucose"],
    bins=20,
    kde=True
)

plt.title("Glucose Level Distribution")
plt.xlabel("Glucose")
plt.ylabel("Frequency")

plt.show()

# ============================================
# 3. BMI DISTRIBUTION
# ============================================

plt.figure(figsize=(7, 4))

sns.histplot(
    data["BMI"],
    bins=20,
    kde=True
)

plt.title("BMI Distribution")
plt.xlabel("BMI")
plt.ylabel("Frequency")

plt.show()

# ============================================
# 4. AGE DISTRIBUTION
# ============================================

plt.figure(figsize=(7, 4))

sns.histplot(
    data["Age"],
    bins=20,
    kde=True
)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")

plt.show()

# ============================================
# 5. CORRELATION HEATMAP
# ============================================

plt.figure(figsize=(10, 7))

correlation = data.corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Feature Correlation Heatmap")

plt.show()

print("\nEDA completed successfully!")
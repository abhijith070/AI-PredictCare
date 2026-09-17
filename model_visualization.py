import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    roc_curve,
    roc_auc_score
)


# ============================================
# LOAD DATASET
# ============================================

data = pd.read_csv("data/diabetes.csv")

# Replace invalid zero values
columns_to_clean = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

for column in columns_to_clean:
    data[column] = data[column].replace(
        0,
        pd.NA
    )

    data[column] = data[column].fillna(
        data[column].median()
    )


# ============================================
# FEATURES AND TARGET
# ============================================

X = data.drop(
    "Outcome",
    axis=1
)

y = data["Outcome"]


# ============================================
# TRAIN TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================
# CREATE MODELS
# ============================================

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000
        ))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            probability=True,
            random_state=42
        ))
    ]),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )
}


# ============================================
# ROC CURVE
# ============================================

plt.figure(figsize=(8, 6))


for name, model in models.items():

    model.fit(
        X_train,
        y_train
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    fpr, tpr, thresholds = roc_curve(
        y_test,
        probabilities
    )

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {auc:.3f})"
    )


# Random classifier reference line

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)


plt.title("ROC Curve Comparison")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.legend()

plt.grid()

plt.show()


# ============================================
# COMPLETED
# ============================================

print("\nROC curve generated successfully!")
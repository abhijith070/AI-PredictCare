import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# ============================================
# LOAD DATASET
# ============================================

data = pd.read_csv(
    "data/heart_clean.csv"
)

print("Heart dataset loaded successfully!")

print(
    "Dataset shape:",
    data.shape
)


# ============================================
# FEATURES AND TARGET
# ============================================

X = data.drop(
    "target",
    axis=1
)

y = data["target"]


print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print("target")


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

print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# ============================================
# CREATE MODELS
# ============================================

models = {

    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(
            strategy="median"
        )),

        ("scaler", StandardScaler()),

        ("classifier", LogisticRegression(
            max_iter=1000
        ))
    ]),


    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(
            strategy="median"
        )),

        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ))
    ]),


    "SVM": Pipeline([
        ("imputer", SimpleImputer(
            strategy="median"
        )),

        ("scaler", StandardScaler()),

        ("classifier", SVC(
            probability=True,
            random_state=42
        ))
    ]),


    "Gradient Boosting": Pipeline([
        ("imputer", SimpleImputer(
            strategy="median"
        )),

        ("classifier", GradientBoostingClassifier(
            random_state=42
        ))
    ])
}


# ============================================
# TRAIN AND EVALUATE MODELS
# ============================================

results = []

trained_models = {}


for name, model in models.items():

    print(
        "\nTraining:",
        name
    )

    # Train
    model.fit(
        X_train,
        y_train
    )

    # Prediction
    predictions = model.predict(
        X_test
    )

    # Probability
    probabilities = model.predict_proba(
        X_test
    )[:, 1]


    # Metrics
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )


    # Store results
    results.append({

        "Model": name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1,

        "ROC-AUC": roc_auc
    })


    trained_models[name] = model


# ============================================
# MODEL COMPARISON
# ============================================

results_df = pd.DataFrame(
    results
)


print("\n================================")
print("MODEL COMPARISON")
print("================================")

print(
    results_df.round(3).to_string(
        index=False
    )
)


# ============================================
# SELECT BEST MODEL
# ============================================

best_model_name = results_df.loc[
    results_df["ROC-AUC"].idxmax(),
    "Model"
]


best_model = trained_models[
    best_model_name
]


print("\n================================")
print("BEST MODEL")
print("================================")

print(
    "Selected model:",
    best_model_name
)


# ============================================
# CREATE MODEL FOLDER
# ============================================

import os

os.makedirs(
    "model",
    exist_ok=True
)


# ============================================
# SAVE MODEL
# ============================================

joblib.dump(
    best_model,
    "model/heart_model.pkl"
)

print(
    "\nHeart disease model saved successfully!"
)


# ============================================
# FINAL MODEL PERFORMANCE
# ============================================

final_predictions = best_model.predict(
    X_test
)

final_probabilities = best_model.predict_proba(
    X_test
)[:, 1]


print("\n================================")
print("MODEL PERFORMANCE")
print("================================")

print(
    "Accuracy :",
    round(
        accuracy_score(
            y_test,
            final_predictions
        ),
        3
    )
)

print(
    "Precision:",
    round(
        precision_score(
            y_test,
            final_predictions
        ),
        3
    )
)

print(
    "Recall   :",
    round(
        recall_score(
            y_test,
            final_predictions
        ),
        3
    )
)

print(
    "F1 Score :",
    round(
        f1_score(
            y_test,
            final_predictions
        ),
        3
    )
)

print(
    "ROC-AUC  :",
    round(
        roc_auc_score(
            y_test,
            final_probabilities
        ),
        3
    )
)


# ============================================
# CLASSIFICATION REPORT
# ============================================

print("\n================================")
print("CLASSIFICATION REPORT")
print("================================")

print(
    classification_report(
        y_test,
        final_predictions
    )
)


# ============================================
# CONFUSION MATRIX
# ============================================

print("\n================================")
print("CONFUSION MATRIX")
print("================================")

print(
    confusion_matrix(
        y_test,
        final_predictions
    )
)
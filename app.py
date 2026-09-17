import streamlit as st
import pandas as pd
import joblib
import shap


# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="AI PredictCare",
    page_icon="🩺",
    layout="wide"
)


# ============================================
# LOAD MODELS
# ============================================

diabetes_model = joblib.load(
    "model/diabetes_model.pkl"
)

heart_model = joblib.load(
    "model/heart_model.pkl"
)


# ============================================
# HEADER
# ============================================

st.title("🩺 AI PredictCare")

st.subheader(
    "AI-Based Health Risk Prediction System"
)

st.write(
    "Select a health condition and enter the "
    "required information to obtain an "
    "ML-based risk prediction."
)


# ============================================
# SELECT PREDICTION
# ============================================

st.sidebar.header("Prediction Type")

prediction_type = st.sidebar.selectbox(
    "Select a condition",
    [
        "Diabetes",
        "Heart Disease"
    ]
)


# ============================================
# DIABETES PREDICTION
# ============================================

if prediction_type == "Diabetes":

    st.header("🩸 Diabetes Risk Prediction")

    st.write(
        "Enter the required health parameters."
    )

    st.divider()


    # ========================================
    # DIABETES INPUTS
    # ========================================

    col1, col2 = st.columns(2)


    with col1:

        pregnancies = st.number_input(
            "Number of Pregnancies",
            min_value=0,
            max_value=20,
            value=1
        )

        glucose = st.number_input(
            "Glucose Level",
            min_value=0,
            max_value=300,
            value=120
        )

        blood_pressure = st.number_input(
            "Blood Pressure",
            min_value=0,
            max_value=200,
            value=70
        )

        skin_thickness = st.number_input(
            "Skin Thickness",
            min_value=0,
            max_value=100,
            value=20
        )


    with col2:

        insulin = st.number_input(
            "Insulin",
            min_value=0,
            max_value=1000,
            value=80
        )

        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            max_value=70.0,
            value=25.0
        )

        pedigree = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=3.0,
            value=0.5
        )

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=100,
            value=30
        )


    st.divider()


    # ========================================
    # DIABETES PREDICTION BUTTON
    # ========================================

    if st.button(
        "🔍 Predict Diabetes Risk",
        use_container_width=True
    ):

        input_data = pd.DataFrame(
            [[
                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                insulin,
                bmi,
                pedigree,
                age
            ]],
            columns=[
                "Pregnancies",
                "Glucose",
                "BloodPressure",
                "SkinThickness",
                "Insulin",
                "BMI",
                "DiabetesPedigreeFunction",
                "Age"
            ]
        )


        # ====================================
        # PREDICTION
        # ====================================

        prediction = diabetes_model.predict(
            input_data
        )

        probability = diabetes_model.predict_proba(
            input_data
        )[0][1]

        risk_percentage = probability * 100


        # ====================================
        # RESULT
        # ====================================

        st.header("📊 Prediction Result")

        result_col1, result_col2 = st.columns(2)


        with result_col1:

            if prediction[0] == 1:

                st.warning(
                    "⚠️ Higher predicted risk"
                )

            else:

                st.success(
                    "✅ Lower predicted risk"
                )


        with result_col2:

            st.metric(
                "Model-Estimated Probability",
                f"{risk_percentage:.2f}%"
            )


        st.progress(
            float(probability)
        )


        # ====================================
        # SHAP
        # ====================================

        st.divider()

        st.header(
            "🔍 Explainable AI"
        )

        st.write(
            "These features had the greatest "
            "influence on this prediction."
        )


        diabetes_explainer = shap.TreeExplainer(
            diabetes_model
        )

        shap_values = (
            diabetes_explainer.shap_values(
                input_data
            )
        )


        if isinstance(shap_values, list):

            values = shap_values[1][0]

        else:

            values = shap_values[0]


        explanation = pd.DataFrame({

            "Feature": input_data.columns,

            "SHAP Value": values

        })


        explanation["Importance"] = (
            explanation["SHAP Value"].abs()
        )


        explanation = explanation.sort_values(
            "Importance",
            ascending=False
        )


        st.dataframe(
            explanation[
                ["Feature", "SHAP Value"]
            ].head(5),
            hide_index=True,
            use_container_width=True
        )


        # ====================================
        # EDUCATIONAL INSIGHTS
        # ====================================

        st.divider()

        st.header(
            "💡 Educational Health Insights"
        )


        if glucose >= 126:

            st.write(
                "• The entered glucose value is "
                "relatively high. Glucose management "
                "is an important factor in diabetes "
                "risk assessment."
            )


        if bmi >= 25:

            st.write(
                "• The entered BMI is in or above "
                "the overweight range. Maintaining "
                "a healthy weight can support "
                "overall health."
            )


        if blood_pressure >= 80:

            st.write(
                "• The entered blood pressure value "
                "may warrant attention. Blood pressure "
                "should be interpreted using appropriate "
                "clinical measurements."
            )


        if age >= 45:

            st.write(
                "• Age is one of the factors considered "
                "by the machine learning model."
            )


# ============================================
# HEART DISEASE PREDICTION
# ============================================

else:

    st.header("❤️ Heart Disease Risk Prediction")

    st.write(
        "Enter the required cardiovascular "
        "health parameters."
    )

    st.divider()


    # ========================================
    # HEART INPUTS
    # ========================================

    col1, col2 = st.columns(2)


    with col1:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=100,
            value=50
        )

        sex = st.selectbox(
            "Sex",
            [0, 1],
            format_func=lambda x:
                "Female (0)" if x == 0
                else "Male (1)"
        )

        cp = st.selectbox(
            "Chest Pain Type",
            [1, 2, 3, 4]
        )

        trestbps = st.number_input(
            "Resting Blood Pressure",
            min_value=80,
            max_value=220,
            value=120
        )

        chol = st.number_input(
            "Cholesterol",
            min_value=100,
            max_value=600,
            value=200
        )

        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            [0, 1]
        )

        restecg = st.selectbox(
            "Resting ECG",
            [0, 1, 2]
        )


    with col2:

        thalach = st.number_input(
            "Maximum Heart Rate",
            min_value=60,
            max_value=220,
            value=150
        )

        exang = st.selectbox(
            "Exercise-Induced Angina",
            [0, 1]
        )

        oldpeak = st.number_input(
            "ST Depression (Oldpeak)",
            min_value=0.0,
            max_value=7.0,
            value=1.0,
            step=0.1
        )

        slope = st.selectbox(
            "Slope",
            [1, 2, 3]
        )

        ca = st.selectbox(
            "Number of Major Vessels",
            [0, 1, 2, 3]
        )

        thal = st.selectbox(
            "Thalassemia",
            [3, 6, 7]
        )


    st.divider()


    # ========================================
    # HEART PREDICTION BUTTON
    # ========================================

    if st.button(
        "🔍 Predict Heart Disease Risk",
        use_container_width=True
    ):

        heart_input = pd.DataFrame(
            [[
                age,
                sex,
                cp,
                trestbps,
                chol,
                fbs,
                restecg,
                thalach,
                exang,
                oldpeak,
                slope,
                ca,
                thal
            ]],
            columns=[
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
                "thal"
            ]
        )


        # ====================================
        # HEART PREDICTION
        # ====================================

        prediction = heart_model.predict(
            heart_input
        )

        probability = heart_model.predict_proba(
            heart_input
        )[0][1]

        risk_percentage = probability * 100


        # ====================================
        # RESULT
        # ====================================

        st.header("📊 Prediction Result")

        result_col1, result_col2 = st.columns(2)


        with result_col1:

            if prediction[0] == 1:

                st.warning(
                    "⚠️ Higher predicted risk"
                )

            else:

                st.success(
                    "✅ Lower predicted risk"
                )


        with result_col2:

            st.metric(
                "Model-Estimated Probability",
                f"{risk_percentage:.2f}%"
            )


        st.progress(
            float(probability)
        )


        # ====================================
        # SHAP EXPLANATION
        # ====================================

        st.divider()

        st.header(
            "🔍 Explainable AI"
        )

        st.write(
            "These features had the greatest "
            "influence on this prediction."
        )


        # Get the imputer and Random Forest
        imputer = heart_model.named_steps[
            "imputer"
        ]

        classifier = heart_model.named_steps[
            "classifier"
        ]


        # Transform input using the same
        # imputer used during training

        transformed_input = imputer.transform(
            heart_input
        )


        heart_explainer = shap.TreeExplainer(
            classifier
        )


        heart_shap_values = (
            heart_explainer.shap_values(
                transformed_input
            )
        )


        if isinstance(
            heart_shap_values,
            list
        ):

            heart_values = (
                heart_shap_values[1][0]
            )

        else:

            if (
                len(heart_shap_values.shape) == 3
            ):

                heart_values = (
                    heart_shap_values[0, :, 1]
                )

            else:

                heart_values = (
                    heart_shap_values[0]
                )


        heart_explanation = pd.DataFrame({

            "Feature": heart_input.columns,

            "SHAP Value": heart_values

        })


        heart_explanation["Importance"] = (
            heart_explanation[
                "SHAP Value"
            ].abs()
        )


        heart_explanation = (
            heart_explanation.sort_values(
                "Importance",
                ascending=False
            )
        )


        st.dataframe(
            heart_explanation[
                ["Feature", "SHAP Value"]
            ].head(5),
            hide_index=True,
            use_container_width=True
        )


# ============================================
# DISCLAIMER
# ============================================

st.divider()

st.caption(
    "⚠️ Disclaimer: AI PredictCare is an "
    "educational machine learning project. "
    "The predictions are model estimates and "
    "are not medical diagnoses. They should not "
    "replace advice from a qualified healthcare "
    "professional."
)
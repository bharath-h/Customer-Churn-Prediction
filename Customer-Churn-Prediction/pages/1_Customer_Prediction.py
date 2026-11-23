
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction")
st.write(
    "Enter customer details to estimate the likelihood "
    "of customer churn using your trained ML models."
)


# --------------------------------------------------
# Load saved models
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"


@st.cache_resource
def load_models():
    logistic_model = joblib.load(
        MODEL_DIR / "logistic_regression_pipeline.pkl"
    )

    random_forest_model = joblib.load(
        MODEL_DIR / "tuned_random_forest.pkl"
    )

    thresholds = joblib.load(
        MODEL_DIR / "model_thresholds.pkl"
    )

    return logistic_model, random_forest_model, thresholds


try:
    logistic_model, random_forest_model, thresholds = load_models()
except Exception as error:
    st.error(f"Could not load the saved models: {error}")
    st.stop()


# --------------------------------------------------
# Exact feature order used during model training
# --------------------------------------------------
FEATURE_COLUMNS = [
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "PaperlessBilling",
    "MonthlyCharges",
    "TotalCharges",
    "gender_Male",
    "MultipleLines_No phone service",
    "MultipleLines_Yes",
    "InternetService_Fiber optic",
    "InternetService_No",
    "OnlineSecurity_No internet service",
    "OnlineSecurity_Yes",
    "OnlineBackup_No internet service",
    "OnlineBackup_Yes",
    "DeviceProtection_No internet service",
    "DeviceProtection_Yes",
    "TechSupport_No internet service",
    "TechSupport_Yes",
    "StreamingTV_No internet service",
    "StreamingTV_Yes",
    "StreamingMovies_No internet service",
    "StreamingMovies_Yes",
    "Contract_One year",
    "Contract_Two year",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check",
    "TotalServices",
    "TenureGroup_Medium",
    "TenureGroup_Long-term",
    "TenureGroup_Loyal"
]


# --------------------------------------------------
# Customer input form
# --------------------------------------------------
st.subheader("Customer Information")

with st.form("customer_prediction_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Personal Details")

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            ["No", "Yes"]
        )

        partner = st.selectbox(
            "Has Partner?",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Has Dependents?",
            ["Yes", "No"]
        )

    with col2:
        st.markdown("#### Account Details")

        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=72,
            value=12,
            step=1
        )

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    with col3:
        st.markdown("#### Charges")

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=18.25,
            max_value=118.75,
            value=70.35,
            step=0.01
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            max_value=8684.80,
            value=monthly_charges * tenure,
            step=0.01
        )

    st.divider()
    st.subheader("Subscribed Services")

    service_col1, service_col2, service_col3 = st.columns(3)

    with service_col1:
        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No phone service", "No", "Yes"]
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        online_security = st.selectbox(
            "Online Security",
            ["No", "Yes", "No internet service"]
        )

    with service_col2:
        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

    with service_col3:
        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

    st.divider()

    selected_model = st.selectbox(
        "Select Prediction Model",
        ["Logistic Regression", "Random Forest"]
    )

    submitted = st.form_submit_button(
        "Predict Customer Churn",
        type="primary",
        use_container_width=True
    )


# --------------------------------------------------
# Feature engineering and prediction
# --------------------------------------------------
if submitted:

    # Calculate TotalServices using the project logic
    total_services = (
        int(phone_service == "Yes")
        + int(multiple_lines == "Yes")
        + int(internet_service != "No")
        + int(online_security == "Yes")
        + int(online_backup == "Yes")
        + int(device_protection == "Yes")
        + int(tech_support == "Yes")
        + int(streaming_tv == "Yes")
        + int(streaming_movies == "Yes")
    )

    # Recreate tenure group dummy variables
    tenure_medium = int(12 < tenure <= 24)
    tenure_long_term = int(24 < tenure <= 48)
    tenure_loyal = int(tenure > 48)

    # Build the exact 34-feature input row
    customer_features = {
        "SeniorCitizen": int(senior_citizen == "Yes"),
        "Partner": int(partner == "Yes"),
        "Dependents": int(dependents == "Yes"),
        "tenure": tenure,
        "PhoneService": int(phone_service == "Yes"),
        "PaperlessBilling": int(paperless_billing == "Yes"),
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,

        "gender_Male": int(gender == "Male"),

        "MultipleLines_No phone service":
            int(multiple_lines == "No phone service"),
        "MultipleLines_Yes":
            int(multiple_lines == "Yes"),

        "InternetService_Fiber optic":
            int(internet_service == "Fiber optic"),
        "InternetService_No":
            int(internet_service == "No"),

        "OnlineSecurity_No internet service":
            int(online_security == "No internet service"),
        "OnlineSecurity_Yes":
            int(online_security == "Yes"),

        "OnlineBackup_No internet service":
            int(online_backup == "No internet service"),
        "OnlineBackup_Yes":
            int(online_backup == "Yes"),

        "DeviceProtection_No internet service":
            int(device_protection == "No internet service"),
        "DeviceProtection_Yes":
            int(device_protection == "Yes"),

        "TechSupport_No internet service":
            int(tech_support == "No internet service"),
        "TechSupport_Yes":
            int(tech_support == "Yes"),

        "StreamingTV_No internet service":
            int(streaming_tv == "No internet service"),
        "StreamingTV_Yes":
            int(streaming_tv == "Yes"),

        "StreamingMovies_No internet service":
            int(streaming_movies == "No internet service"),
        "StreamingMovies_Yes":
            int(streaming_movies == "Yes"),

        "Contract_One year":
            int(contract == "One year"),
        "Contract_Two year":
            int(contract == "Two year"),

        "PaymentMethod_Credit card (automatic)":
            int(payment_method == "Credit card (automatic)"),
        "PaymentMethod_Electronic check":
            int(payment_method == "Electronic check"),
        "PaymentMethod_Mailed check":
            int(payment_method == "Mailed check"),

        "TotalServices": total_services,
        "TenureGroup_Medium": tenure_medium,
        "TenureGroup_Long-term": tenure_long_term,
        "TenureGroup_Loyal": tenure_loyal
    }

    input_df = pd.DataFrame(
        [customer_features],
        columns=FEATURE_COLUMNS
    )

    # Select model and its saved threshold
    if selected_model == "Logistic Regression":
        model = logistic_model
        threshold = thresholds["logistic_regression"]
    else:
        model = random_forest_model
        threshold = thresholds["random_forest"]

    # Predict churn probability
    probability = model.predict_proba(input_df)[0][1]

    # Apply the saved threshold
    prediction = int(probability >= threshold)

    st.divider()
    st.subheader("Prediction Results")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%"
        )

    with result_col2:
        st.metric(
            "Decision Threshold",
            f"{threshold * 100:.0f}%"
        )

    with result_col3:
        st.metric(
            "Total Services",
            total_services
        )

    st.progress(float(probability))

    if prediction == 1:
        st.error(
            "⚠️ High Churn Risk: The model predicts that "
            "this customer may leave the service."
        )
    else:
        st.success(
            "✅ Low Churn Risk: The model predicts that "
            "this customer is likely to stay."
        )

    st.caption(
        f"Model used: {selected_model} | "
        f"Prediction rule: churn probability ≥ {threshold:.2f}"
    )

    with st.expander("View processed model input"):
        st.dataframe(input_df, use_container_width=True)
        st.write("Number of input features:", len(input_df.columns))

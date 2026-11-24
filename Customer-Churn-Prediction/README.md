# Customer Churn Prediction

A Machine Learning project that predicts whether a telecom customer is likely to leave a service provider. This project uses Logistic Regression and Random Forest models to identify customers at risk of churn and support data-driven customer retention strategies.

## Project Overview

Customer churn is a major challenge for subscription-based businesses. Retaining existing customers is often more cost-effective than acquiring new ones.

This project analyzes customer demographics, account information, subscribed services, and billing details to identify patterns associated with customer churn.

The project includes data preprocessing, exploratory data analysis, feature engineering, model training, evaluation, and an interactive Streamlit application for customer churn prediction.

## Objectives

- Analyze customer data and identify churn patterns.
- Perform data cleaning and exploratory data analysis.
- Engineer relevant features to improve model performance.
- Train and compare multiple machine learning models.
- Evaluate models using classification metrics.
- Predict customer churn using an interactive application.

## Technologies Used

- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn, Plotly
- **Machine Learning:** Scikit-learn
- **Model Persistence:** Joblib
- **Web Application:** Streamlit
- **Development Environment:** Jupyter Notebook, VS Code

## Dataset

**Dataset:** Telco Customer Churn

The dataset contains customer information, including:

- Customer demographics
- Tenure and contract details
- Internet and phone services
- Payment methods
- Monthly and total charges
- Customer churn status

The cleaned dataset contains 7,043 customer records and 20 columns.

## Project Workflow

1. Data Collection
2. Data Understanding
3. Data Cleaning and Preprocessing
4. Exploratory Data Analysis (EDA)
5. Feature Engineering
6. Model Training
7. Model Evaluation
8. Customer Churn Prediction

## Machine Learning Models

The following models were implemented and evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- Tuned Random Forest

The project also explores threshold optimization to improve the detection of customers likely to churn.

## Model Performance

### Default Threshold Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 80.28% | 67.27% | 49.73% | 57.19% | 0.8406 |
| Decision Tree | 78.86% | 65.31% | 43.01% | 51.86% | 0.8212 |
| Random Forest | 79.36% | 65.07% | 47.58% | 54.97% | 0.8335 |

### Threshold Optimization Results

| Model | Threshold | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.35 | 77.15% | 55.63% | 67.74% | 61.09% |
| Random Forest | 0.30 | 75.44% | 52.58% | 73.92% | 61.45% |

**Key Observation:** Logistic Regression achieved the highest ROC-AUC of 0.8406 at the default threshold. After threshold optimization, Random Forest achieved the highest recall of 73.92%, helping identify more potential churn customers.

## Project Structure

```text
Customer-Churn-Prediction/
│
├── dataset/
│   ├── Telco-Customer-Churn.csv
│   ├── Telco-Customer-Churn-Cleaned.csv
│   ├── Telco-Customer-Churn-Feature-Engineered.csv
│   └── Telco-Customer-Churn-Cleaned-v2.csv
│
├── models/
│   ├── feature_importance.csv
│   ├── feature_importance.png
│   ├── logistic_regression_pipeline.pkl
│   ├── model_thresholds.pkl
│   └── tuned_random_forest.pkl
│
├── notebooks/
│   ├── 01_Data_Collection.ipynb
│   ├── 02_Data_Understanding.ipynb
│   ├── 03_Data_Cleaning.ipynb
│   ├── 04_Exploratory_Data_Analysis.ipynb
│   ├── 05_Feature_Engineering.ipynb
│   ├── 06_Model_Training.ipynb
│   └── 07_Model_Evaluation.ipynb
│
├── pages/
│   ├── 1_Customer_Prediction.py
│   ├── 2_Churn_Analytics.py
│   ├── 3_Model_Insights.py
│   └── 4_Batch_Prediction.py
│
├── screenshots/
│   ├── home-1.png
│   ├── home-2.png
│   ├── customer-prediction-1.png
│   ├── customer-prediction-2.png
│   ├── customer-prediction-3.png
│   ├── churn-analytics-1.png
│   ├── churn-analytics-2.png
│   ├── churn-analytics-3.png
│   ├── churn-analytics-4.png
│   ├── churn-analytics-5.png
│   ├── model-insights-1.png
│   ├── model-insights-2.png
│   ├── model-insights-3.png
│   ├── model-insights-4.png
│   ├── model-insights-5.png
│   ├── batch-prediction-1.png
│   ├── batch-prediction-2.png
│   ├── batch-prediction-3.png
│   └── batch-prediction-4.png
│
├── app.py
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/bharath-h/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal to access the application.

## Future Improvements

- Add more advanced machine learning algorithms.
- Improve model performance through hyperparameter optimization.
- Integrate explainable AI techniques for prediction interpretation.
- Develop customer retention recommendations.
- Deploy the application to a cloud platform.

## Author

**Bharath Eslavath**

- GitHub: [bharath-h](https://github.com/bharath-h)
- Project Repository: [Customer-Churn-Prediction](https://github.com/bharath-h/Customer-Churn-Prediction)

---

If you find this project useful, consider giving the repository a star.

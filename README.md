# Telco Customer Churn Analysis & Prediction

## Project Overview
Welcome to the **Telco Customer Churn Analysis** project! 

In the telecommunications industry, retaining customers is often more cost-effective than acquiring new ones. **Customer Churn** happens when a customer stops doing business with a company. This project dives deep into a dataset of customer profiles to understand *why* they leave and builds machine learning models to **predict** who is at risk of churning.

**Goal:** Identify at-risk customers early so the business can take proactive retention measures.

---

## The Data
We are using the [Telco Customer Churn dataset from Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/).


| Column Name | Description |
| :--- | :--- |
| `customerID` | Customer ID |
| `gender` | Whether the customer is a male or a female |
| `SeniorCitizen` | Whether the customer is a senior citizen or not (1, 0) |
| `Partner` | Whether the customer has a partner or not (Yes, No) |
| `Dependents` | Whether the customer has dependents or not (Yes, No) |
| `tenure` | Number of months the customer has stayed with the company |
| `PhoneService` | Whether the customer has a phone service or not (Yes, No) |
| `MultipleLines` | Whether the customer has multiple lines or not (Yes, No, No phone service) |
| `InternetService` | Customer’s internet service provider (DSL, Fiber optic, No) |
| `OnlineSecurity` | Whether the customer has online security or not (Yes, No, No internet service) |
| `OnlineBackup` | Whether the customer has online backup or not (Yes, No, No internet service) |
| `DeviceProtection` | Whether the customer has device protection or not (Yes, No, No internet service) |
| `TechSupport` | Whether the customer has tech support or not (Yes, No, No internet service) |
| `StreamingTV` | Whether the customer has streaming TV or not (Yes, No, No internet service) |
| `StreamingMovies` | Whether the customer has streaming movies or not (Yes, No, No internet service) |
| `Contract` | The contract term of the customer (Month-to-month, One year, Two year) |
| `PaperlessBilling` | Whether the customer has paperless billing or not (Yes, No) |
| `PaymentMethod` | The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)) |
| `MonthlyCharges` | The amount charged to the customer monthly |
| `TotalCharges` | The total amount charged to the customer |
| `Churn` | Whether the customer churned or not (Yes or No) |

---

## Project Workflow

Our approach is structured to go from raw data to a production-ready model:

### 1. Exploratory Data Analysis (EDA)
We started by getting to know the data. Key insights included:
*   **Tenure:** Newer customers are much more likely to churn.
*   **Monthly Charges:** Higher monthly bills correlate with higher churn rates.
*   **Total Charges:** Highly correlated with tenure and monthly charges (Multicollinearity alert!).
*   **Imbalance:** The dataset is imbalanced (more people stayed than left), which we addressed during modeling.

### 2. Data Preprocessing
To prepare the data for our models, we performed:
*   **Cleaning:** Handled missing values in `TotalCharges`.
*   **Encoding:** Used **Label Encoding** for categorical variables.
*   **Feature Selection:** 
    *   Used **VIF (Variance Inflation Factor)** to identify multicollinearity. We dropped `TotalCharges` because it was redundant with `Tenure` and `MonthlyCharges`.
    *   Used **Chi-Squared Test** to drop features with low statistical significance (`gender`, `PhoneService`).
*   **Splitting:** 80% Training / 20% Testing (Stratified to maintain churn proportion).
*   **Scaling:** Applied `StandardScaler` to numerical features to ensure fair comparisons between models.

### 3. Modeling Strategy
We implemented a robust two-stage modeling process:
1.  **Baseline Testing:** We trained 5 different models with default settings to establish benchmarks.
    *   Logistic Regression
    *   Random Forest
    *   K-Nearest Neighbors (KNN)
    *   XGBoost
    *   LightGBM
2.  **Hyperparameter Tuning:** We selected the top performers and optimized them using a combination of **RandomizedSearchCV** (broad search) and **GridSearchCV** (fine-tuning).

---

## Results & Conclusion

Predicting churn is tricky! We focused on **AUC-ROC** (overall performance) and **Recall** (catching as many churners as possible).

| Model | Test Accuracy | Test AUC-ROC | Test Recall (Churn) |
| :--- | :--- | :--- | :--- |
| **XGBoost (Optimized)** | **84.6%** | **0.846** | 51.6% |
| **LightGBM (Optimized)** | 84.4% | 0.844 | **81.3%** |
| Logistic Regression (Opt) | 83.3% | 0.833 | 52.4% |
| Random Forest (Baseline) | 77.1% | 0.815 | 47.1% |
| KNN (Baseline) | 74.5% | 0.751 | 50.0% |

### Conclusion & Winner

After exploring the data, building baselines, and carefully tuning three promising models, we have reliable churn predictors ready for real-world use.

**The winner is: Optimized LightGBM**
Yes, XGBoost has a slightly higher AUC-ROC (0.846 vs. 0.844), but in customer churn prediction, recall on the churn class is often the most critical metric and LightGBM dominates here with an impressive 0.813 recall.


### Why recall matters most in churn prediction

- **Business cost asymmetry:** Missing a customer who is about to churn (false negative) is far more expensive than incorrectly flagging a loyal customer (false positive).
    - If we fail to identify a churner, we lose that customer's lifetime value, potentially thousands of dollars in recurring revenue.
    - If we mistakenly target a non-churner with a retention offer (discount, free month, etc.), the cost is usually small and temporary.

- Proactive retention: The whole point of churn modeling is to intervene early. High recall means we catch significantly more at-risk customers, giving the company more opportunities to save them through personalized offers, better support, or service improvements.
- Real-world impact: LightGBM's 0.813 recall means it identifies ~81% of actual churners, that's ~30% more than XGBoost's 0.516 in relative terms. Even with a tiny drop in precision, the extra saved customers easily justify the choice.

LightGBM also delivers a strong F1-score (0.617) and very competitive AUC/accuracy, showing it's not sacrificing too much overall performance for that recall boost.

Runner-up: Optimized XGBoost remains excellent if the business wants the absolute best discrimination (AUC) with fewer false positives.

**Final recommendation**
Deploy the Optimized LightGBM model for maximum retention impact. Pair it with SHAP explanations to understand why each customer is flagged, enabling targeted and effective interventions.

Thanks for following this analysis! This model can genuinely help a telco reduce churn and protect revenue. Feel free to experiment further, maybe try class weights, SMOTE, or ensembling the two top models for an even sweeter spot.

---

## How to Run
1.  Clone this repository.
2.  Install the required libraries:
```bash
pip install -r requirements.txt
```
3.  Run the notebook `customer-churn-analysis-and-prediction.ipynb`.

---

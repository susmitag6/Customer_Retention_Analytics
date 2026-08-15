# Customer Churn & Retention Analytics Dashboard

## About this project

This is an end-to-end customer analytics project built using the IBM Telco Customer Churn dataset.

The goal is simple: identify customers who may leave, understand different customer groups, estimate which customers are valuable to retain, and show the results in one dashboard.

## What the project does

The project:

1. Cleans the raw customer data.
2. Explores customer churn patterns.
3. Trains a machine learning model to predict churn.
4. Compares Logistic Regression and Random Forest.
5. Optimizes the churn classification threshold.
6. Groups customers using K-Means clustering and PCA for cluster visualization in 2D
7. Combines churn probability with Customer Lifetime Value (CLTV).
8. Creates a retention priority score.
9. Displays the results in a Streamlit dashboard.

## Dataset

The project uses the IBM Telco Customer Churn dataset.

The dataset contains 7,043 customers and information such as:

- tenure
- monthly charges
- total charges
- contract type
- internet service
- payment method
- customer services
- churn status
- CLTV

The target variable is `churn_value`:

- `0` = customer stayed
- `1` = customer churned

## Project workflow

```text
Raw Telco Data
      |
      v
Data Cleaning
      |
      v
Exploratory Analysis
      |
      v
Feature Preparation
      |
      v
Churn Prediction
      |
      +---- Logistic Regression
      |
      +---- Random Forest (Further accuracy can be improved)
      |
      v
Threshold Optimization
      |
      v
Customer Segmentation + principal comp. (K-Means + PCA)
      |
      v
Churn Probability + CLTV
      |
      v
Retention Priority
      |
      v
Streamlit Dashboard
```

## Churn prediction

I tested Logistic Regression and Random Forest.

Logistic Regression results at the original 0.50 threshold:

| Metric | Score |
|---|---:|
| Accuracy | 80.20% |
| Precision | 64.35% |
| Recall | 56.95% |
| F1 Score | 60.43% |
| ROC-AUC | 84.87% |

Random Forest had a slightly higher ROC-AUC, but Logistic Regression was retained because it was competitive and easier to interpret.

### Threshold optimization

For a churn problem, missing a customer who is actually going to leave can be costly.

I tested several classification thresholds and selected `0.35`.

At this threshold:

| Metric | Score |
|---|---:|
| Precision | 56.07% |
| Recall | 71.66% |
| F1 Score | 62.91% |

The lower threshold improves recall, so the model catches more actual churners.

## Customer risk groups

Customers are grouped using their predicted churn probability:

- **Low Risk:** below 35%
- **Medium Risk:** 35% to below 60%
- **High Risk:** 60% or higher

Current distribution:

| Risk | Customers |
|---|---:|
| Low | 4,690 |
| Medium | 1,208 |
| High | 1,145 |

## Customer segmentation

I used K-Means clustering to create customer segments.

K=2 had the best silhouette score, but K=4 was selected for the final analysis because the four groups were more useful for business interpretation.

The final segments are:

- **Established High-Value**
- **Low-Spend Stable**
- **New High-Risk**
- **Mid-Tenure Moderate-Risk**

One of the strongest findings was the New High-Risk segment, which had a historical churn rate of about 46.85%.

## Retention priority

Churn probability alone does not tell the business which customer should be contacted first.

I created a simple value-at-risk measure:

```text
Value at Risk = Churn Probability × CLTV
```

Customers are then ranked into:

- Critical
- High
- Medium
- Low

Current priority distribution:

| Priority | Customers | Avg. Churn Probability | Avg. CLTV |
|---|---:|---:|---:|
| Critical | 705 | 68.94% | 5,068.31 |
| High | 996 | 57.13% | 4,135.94 |
| Medium | 652 | 50.91% | 2,869.56 |
| Low | 4,690 | 10.33% | 4,568.82 |

This helps answer a more useful business question:

> Which customers have both meaningful churn risk and meaningful customer value?

## Dashboard

The Streamlit dashboard includes:

- total customers
- actual churn rate
- high-risk customers
- critical retention customers
- actionable value at risk
- churn-risk distribution
- customer-segment distribution
- retention-priority distribution
- churn risk by segment
- CLTV vs. churn probability
- top retention opportunities
- PCA visulization for K-Means customer segments
- filters for risk, segment, priority, and contract type

## Project structure

```text
customer_analytics_dashboard_project/
|
|-- data/
|   |-- raw/
|   |-- processed/
|   `-- model_output/
|
|
|-- notebooks/eda.ipynb
|
|-- models/
|   `-- churn_pipeline.pkl
|
|-- src/
|   |-- data_cleaning.py
|   |-- eda.py
|   |-- model_comparison.py
|   |-- threshold_optimization.py
|   |-- train_model.py
|   |-- score_customers.py
|   |-- final_segmentation.py
|   |-- customer_analytics.py
|   `-- retention_priority.py
|
|-- dashboard/
|   `-- app.py
|
`-- requirements.txt
```

## How to run

Install the required packages:

```bash
pip install -r requirements.txt
```

Train the churn model:

```bash
python src/train_model.py
```

Score customers:

```bash
python src/score_customers.py
```

Run the segmentation and analytics scripts as needed, then start the dashboard:

```bash
streamlit run dashboard/app.py
```
<<<<<<< 
🌐 **Live Demo:** [customerretentionanalytics-ajwsejzc9lqsatn8sc28jg.streamlit.app/](//https://customerretentionanalytics-ajwsejzc9lqsatn8sc28jg.streamlit.app//)

## Main tools

- Python
- Pandas
- NumPy
- Scikit-learn
- Logistic Regression
- Random Forest
- K-Means
- Plotly
- Streamlit
- Joblib

## Main business insights

- Shorter-tenure customers show stronger churn risk.
- Month-to-month contracts are associated with higher churn.
- Two-year contracts are associated with lower churn.
- The New High-Risk segment has particularly high churn.
- Using CLTV together with churn probability gives a better retention priority than churn probability alone.


## Final takeaway

This project is not only about predicting churn.

The main goal is to turn a machine learning prediction into a business decision:

**Who is likely to leave, which customers matter most, and who should the retention team contact first?**

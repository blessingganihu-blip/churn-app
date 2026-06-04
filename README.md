# Netflix Customer Churn Prediction

## Project Overview
This project builds a machine learning model to predict customer churn 
for a Netflix-like streaming service. The goal is to identify customers 
who are likely to cancel their subscription so the business can take 
proactive retention measures.

## Dataset
- Source: Kaggle
- 5,000 customer records
- Features include: watch hours, last login days, subscription type, 
  number of profiles, average watch hours per day, payment method, 
  favorite genre, region and device.

## Models Used
| Model | Accuracy |
|-------|----------|
| Random Forest | 98% |
| Logistic Regression | 90% |

✅ Random Forest was selected as the final model due to its superior performance.

## Key Findings
- `last_login_days` was the strongest predictor of churn
- Customers inactive for 30-60 days had a 75% churn rate
- Customers active within 7 days had only a 14.8% churn rate
- The dataset was perfectly balanced (50/50 churn vs non-churn)

## Project Structure# churn-app

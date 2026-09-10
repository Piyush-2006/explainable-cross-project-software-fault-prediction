# Real-Time Explainable AI-Based Software Fault Detection and Risk Monitoring

## 1. Problem Statement

Modern software projects contain a large number of modules, making it
difficult for development and testing teams to test every module with
equal priority.

Defects are more likely to occur in modules with high complexity, code
churn, coupling, large code size, and historical defects.

This project proposes an AI-based system that predicts the defect risk of
software modules and provides explanations for the predictions.

## 2. Objective

The main objective is to develop a real-time software fault prediction
system that:

- Analyzes software metrics.
- Predicts module-level defect probability.
- Classifies modules into low, medium, and high risk.
- Explains predictions using Explainable AI.
- Prioritizes software testing.
- Processes new software changes in real time.

## 3. Machine Learning

The project will compare:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

## 4. Explainable AI

SHAP will be used to identify the important factors influencing each
software fault prediction.

## 5. Real-Time Architecture

The final system will integrate:

GitHub → Webhook → Kafka → Code Analysis → ML → SHAP → Risk Assessment → Dashboard

## 6. Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- GitHub
- Apache Kafka
- FastAPI
- PostgreSQL
- React
- Docker

## 7. Expected Outcome

The system will provide software developers and testers with real-time
information about potentially fault-prone software modules and explain
why those modules are considered risky.
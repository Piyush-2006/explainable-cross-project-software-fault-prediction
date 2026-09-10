# Explainable Cross-Project Software Fault Prediction and Risk-Based Testing

An explainable machine learning system for predicting software fault risk at the module level, prioritizing testing, and analyzing model explanations across software projects.

## 🚀 Live Demo

👉 **[Open Live Demo](https://explainable-cross-project-software-fault.onrender.com/)**

> The free Render instance may take a short time to wake up after inactivity.

---

## 📌 Overview

Software projects contain a large number of modules, making it difficult for testing teams to test every module with equal priority.

This project develops an **Explainable AI-based software fault prediction framework** that:

- Predicts the probability of software defects.
- Converts predictions into a 0–100 risk score.
- Classifies modules as LOW, MEDIUM, or HIGH risk.
- Prioritizes software testing based on predicted risk.
- Explains predictions using SHAP.
- Handles class imbalance using SMOTE.
- Evaluates Cross-Project Defect Prediction (CPDP).
- Measures explanation consistency across projects.
- Provides a FastAPI prediction API.
- Provides an interactive web dashboard.
- Runs using Docker and is deployed on Render.

---

## 🎯 Objectives

The main objectives of this project are:

1. Analyze software metrics associated with defective modules.
2. Clean and preprocess software defect datasets.
3. Compare different machine learning algorithms.
4. Handle class imbalance using SMOTE.
5. Optimize the classification threshold using validation data.
6. Predict module-level defect probability.
7. Generate module-level risk scores.
8. Prioritize software testing.
9. Explain predictions using SHAP.
10. Evaluate Cross-Project Defect Prediction.
11. Analyze explanation consistency across CPDP directions.
12. Deploy the prediction system through an API and web dashboard.

---

## 🔬 Research Questions

### RQ1
How effectively can the proposed approach predict defective software modules within a project?

### RQ2
How does the direction of Cross-Project Defect Prediction affect prediction performance?

### RQ3
How consistent are SHAP feature explanations across different cross-project prediction directions?

---

## 📊 Datasets

The project uses software defect datasets from the NASA/PROMISE software defect dataset collection.

### Projects Used

- **CM1**
- **JM1**

The datasets contain software metrics related to:

- Lines of code
- Cyclomatic complexity
- Halstead metrics
- Operators and operands
- Branches
- Comments
- Code structure

The target variable represents whether a software module contains a defect.

---

## 🧹 Data Preprocessing

The preprocessing pipeline includes:

```text
Raw Dataset
     ↓
Remove ID
     ↓
Convert Defect Labels
     ↓
Handle Invalid Values
     ↓
Remove Duplicate Rows
     ↓
Prepare Features
     ↓
ML-Ready Dataset

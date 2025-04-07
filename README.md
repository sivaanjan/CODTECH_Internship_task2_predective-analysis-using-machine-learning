# CODTECH_task-2_PREDICTIVE-ANALYSIS-USING-MACHINE-LEARNING
# CodTech Internship - Task 2

This repository contains my solution for **Task 2** of the CodTech Internship in the **Data Analysis** domain.

---

## 📌 Task Objective

To perform classification using **Logistic Regression** on the **UCI Adult Income Dataset**, predicting whether a person's income exceeds $50K/year based on demographic attributes.

---

## 📂 Files

- `codtech_task2.py` – Python script containing data preprocessing, modeling, and evaluation
- `adult.data` – Dataset file downloaded from the UCI Machine Learning Repository

---

## 🧪 Task Workflow

1. **Load the Dataset**  
   The data is loaded from a local file (`adult.data`)
   
3. **Preprocessing**  
   - Missing value handling  
   - Dropping irrelevant columns (`fnlwgt`, `education`, `native_country`)  
   - Encoding categorical variables (binary and one-hot)

4. **Model Training**  
   - Data split into training and test sets  
   - Logistic Regression model trained on the dataset

5. **Evaluation**  
   - Accuracy score  
   - Confusion matrix  
   - Classification report (precision, recall, F1-score)  
   - Heatmap visualization of confusion matrix

---

## 🔧 Tools & Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## 📈 Results

- **Model**: Logistic Regression
- **Target Variable**: `income`
- **Accuracy Achieved**: ~[Insert Final Accuracy]
- **Evaluation**: Confusion Matrix + Classification Report

---

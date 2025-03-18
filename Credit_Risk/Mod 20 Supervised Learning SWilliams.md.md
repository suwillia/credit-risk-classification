# Module 20 Supervised Learning
Author: Sunil Williams

## Overview of the Analysis

**Purpose of the analysis** - The purpose  of this analysis is to determine in the Logistic Regression machine learning model can more accurately predict Healthy loans versus High risk loans by using a data set that is resampled to increase the size of the minority class.
**The DataSet** - contains financial information used to predict credit health through loan status classification. It consists of 77,536 records with 8 columns:
**Dataset Structure**
-   **loan_size**: Float - The amount of the loan
-   **interest_rate**: Float - The interest rate charged on the loan
-   **borrower_income**: Integer - The borrower's reported income
-   **debt_to_income**: Float - The ratio of debt to income
-   **num_of_accounts**: Integer - Number of accounts the borrower has
-   **derogatory_marks**: Integer - Count of negative marks on credit report
-   **total_debt**: Integer - Total debt amount
-   **loan_status**: Integer - Target variable (0 = healthy loan, 1 = high risk of default)- This is the result we trained our data 				   to calculate.

**Process **
Data Preparation:
Loaded lending data from CSV file
Split data into features (X) and target variable (y)
Performed train-test split with random_state=1
Created and trained a LogisticRegression model on the imbalanced data
Evaluated performance using balanced_accuracy_score
Generated confusion matrix 
Applied RandomOverSampler to balance the training data
Resampled the minority class (high-risk loans) to match the majority class
Trained a new LogisticRegression model with the resampled data
Evaluated with balanced_accuracy_score
Generated a new confusion matrix 

**Machine Learning methods and supporting functions used**
*Models*
LogisticRegression
RandomOverSampler

*Model training and Evaluation*
train_test_split
model.fit
model.predict

*Performance Metrics*
balanced_accuracy_score
confusion_matrix
classification_report

*Resampling:*
RandomOverSampler.fit_resample
pd.Series().nunique


## Results

 **Original Logistic Regression Model (Imbalanced Data)**

-   **Balanced Accuracy Score**: 0.968
-   **Precision Scores**:
    -   Healthy Loans (0): 1.00
    -   High-Risk Loans (1): 0.84
-   **Recall Scores**:
    -   Healthy Loans (0): 0.99
    -   High-Risk Loans (1): 0.94
-   **F1 Scores**:
    -   Healthy Loans (0): 1.00
    -   High-Risk Loans (1): 0.89

**Logistic Regression Model with RandomOverSampler**

-   **Balanced Accuracy Score**: 0.994
-   **Precision Scores**:
    -   Healthy Loans (0): 1.00
    -   High-Risk Loans (1): 0.84
-   **Recall Scores**:
    -   Healthy Loans (0): 0.99
    -   High-Risk Loans (1): 0.99
-   **F1 Scores**:
    -   Healthy Loans (0): 1.00
    -   High-Risk Loans (1): 0.91

## Summary

**Performance Comparison**

*Original Logistic Regression Model:*
Balanced Accuracy: **0.968**
High-Risk Loan Prediction: Good precision (0.84) but lower recall (0.94)
Healthy Loan Prediction: Excellent precision (1.00) and recall (0.99)

*Logistic Regression with RandomOverSampler:*
Balanced Accuracy: **0.994 (improved)**
High-Risk Loan Prediction: Same precision (0.84) but improved recall (0.99)
Healthy Loan Prediction: Maintained excellent precision (1.00) and recall (0.99)

The oversampled model achieved a great balanced accuracy, indicating better overall performance across both classes. Additionally the oversampled model improved recall for high-risk loans from 0.94 to 0.99, meaning it identifies nearly all high-risk loans. Finally  the oversampled model improved the F1-score for high-risk loans showing better balance between precision and recall.

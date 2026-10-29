# Customer Churn Prediction

A beginner-friendly Streamlit project for exploring customer churn, comparing two machine learning models, and estimating churn for individual customers or customer CSV files. The app uses the public IBM Telco Customer Churn dataset by default.

## Project overview

The app shows customer and churn summaries, compares Logistic Regression with Random Forest, and supports single-customer and batch predictions. Logistic Regression remains the model used for predictions.

## Dataset

The default dataset is `data/ibm_telco_churn.csv`, a public IBM Telco Customer Churn CSV with 7,043 customer rows and 21 source columns. The [IBM dataset repository](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv) is the source. The raw file is not changed by the app.

The app renames these source columns and converts the churn label:

| App column | IBM source column | Use |
| --- | --- | --- |
| `customer_id` | `customerID` | Customer identifier; not a model feature. |
| `tenure_months` | `tenure` | Numeric feature, in months. |
| `monthly_charges` | `MonthlyCharges` | Numeric feature. |
| `contract` | `Contract` | Category feature. |
| `internet_service` | `InternetService` | Category feature. |
| `churned` | `Churn` | Target: `Yes` becomes `1`; `No` becomes `0`. |

The four model features are tenure, monthly charges, contract, and internet service. `TotalCharges` is not used. Its 11 whitespace-only cells are retained in the raw dataset; all correspond to zero-tenure rows. The app also does not use `TechSupport` as a support-ticket count.

`data/customers.csv` remains in the project as an older 20-row fictional sample. It is not the app's default dataset.

## App features

- Explore customer records, customer totals, churn rate, and churn by contract.
- Compare Logistic Regression and Random Forest on the same held-out customers.
- View Accuracy, Precision, Recall, F1-score, ROC-AUC, ROC curves, and a confusion matrix for each model.
- Inspect individual Logistic Regression test predictions and its learned feature associations.
- Make a single-customer prediction or upload a batch CSV and download the results.
- Upload a labeled CSV to use as training data.

## Model comparison results

These are the reported results from the current 75/25 stratified split with `random_state=42` and a fixed 50% classification threshold:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 78% | 60% | 49% | 54% | 84% |
| Random Forest | 76% | 55% | 47% | 51% | 78% |

The numbers are rounded. They describe this dataset and this one split; they do not guarantee future performance.

## How evaluation works

The app separates 25% of the rows for testing and uses the other 75% for training. Stratification keeps the churn/stay proportions similar in both groups. Both models use the exact same split and the same four features. Each model has its own preprocessing pipeline, fitted on training rows only. The test rows are held aside for evaluation.

For the comparison, a predicted churn probability of 50% or higher is labeled churn; a lower probability is labeled no churn. The threshold is fixed for this comparison, not selected by repeatedly checking test scores. The Make prediction page has a separate slider for choosing how individual or batch predictions are labeled.

- **Accuracy:** share of all predictions that were correct.
- **Precision:** share of predicted churners who actually churned.
- **Recall:** share of actual churners the model identified.
- **F1-score:** a combined measure of precision and recall.
- **ROC-AUC:** how well the model ranks churners above customers who stayed, across possible thresholds.
- **Confusion matrix:** counts of correct and incorrect predictions, with actual classes in rows and predicted classes in columns.

## Installation on Windows

1. Install Python 3.10 or newer from [python.org](https://www.python.org/downloads/). During setup, select **Add Python to PATH**.
2. Open PowerShell in the project folder.
3. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use Command Prompt and run:

   ```bat
   .venv\Scripts\activate.bat
   ```

4. Install the packages:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

## Run the app

From the project folder with the virtual environment active:

```powershell
streamlit run app.py
```

Streamlit prints a local web address and usually opens the app in your browser. Choose **Explore data**, **Train model**, or **Make prediction** from the navigation buttons.

## Use another labeled training CSV

A labeled file uploaded in the sidebar must include these columns:

`customer_id`, `tenure_months`, `monthly_charges`, `contract`, `internet_service`, and `churned`.

Use `0` for customers who stayed and `1` for customers who churned. Include at least 8 rows and at least two rows for each outcome. Required values cannot be blank. The app checks the upload and returns to the built-in IBM dataset if the file is invalid.

## Make predictions

For a single customer, enter tenure, monthly charges, contract, and internet service. For batch predictions, download the CSV template and provide `customer_id`, `tenure_months`, `monthly_charges`, `contract`, and `internet_service`. A prediction file does not include `churned`. Logistic Regression is fitted from the selected labeled dataset for these predictions.

## Limitations

- The evaluation is one train/test split, so scores may vary with a different dataset or split.
- The model results are for learning and are not suitable as the sole basis for real customer decisions.
- The IBM dataset may not represent a current company's customers or future behavior.
- The app does not use `TotalCharges`; its missing values have not been filled or changed in the source CSV.

## Project files

```text
Customer_Churn_Prediction/
|-- app.py
|-- requirements.txt
|-- README.md
`-- data/
    |-- customers.csv
    `-- ibm_telco_churn.csv
```
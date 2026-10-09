# Interview Preparation: AI Anomaly Agent

## 30-second project explanation
"I built a Streamlit application that helps analysts screen business datasets for unusual records. Users upload a CSV, inspect data-quality metrics, select numeric features, and run an Isolation Forest model. The app visualizes the results, ranks flagged records, exports them, and generates an investigation brief. The flags are intended to prioritize human review, not to automatically label fraud."

## Explain the architecture
- `app.py`: Streamlit UI, user inputs, charts, and download controls.
- `src/data_quality.py`: missing values, duplicates, column types, unique counts.
- `src/anomaly.py`: preprocessing pipeline and Isolation Forest.
- `src/reporting.py`: transparent rule-based report.
- `src/sample_data.py`: reproducible synthetic ecommerce dataset.
- `tests/`: smoke tests for core logic.

## Concepts to learn
**Anomaly:** A record that differs from the dominant pattern in the selected data.

**Unsupervised learning:** The model does not require pre-labelled anomaly examples to run.

**Isolation Forest:** Builds random partitioning trees. Unusual points tend to be isolated in fewer splits than common points.

**Median imputation:** Replaces missing numeric values with the column median; less sensitive to extreme values than the mean.

**Standardization:** Transforms features to comparable scales. It is included as a reusable preprocessing step; Isolation Forest is tree-based and usually less scale-sensitive than distance-based algorithms.

**Contamination:** A parameter that influences the threshold used to label outliers. It is not a real-world measured anomaly rate.

**False positive:** A normal/valid record flagged as unusual.

**False negative:** A real anomaly that the detector fails to flag.

## Common interview questions

### Why did you choose Isolation Forest?
It is a practical unsupervised baseline for tabular anomaly screening and does not require labelled examples. It is efficient for many datasets, but I would compare it with domain rules, robust statistical methods, and other models.

### Does an anomaly mean fraud?
No. It only means the row looks unusual relative to selected features and the dataset. It may be a valid rare transaction, a data-entry error, or an operational issue. A human must verify it.

### How would you evaluate it?
With labels, I would calculate precision, recall, F1, and a confusion matrix, paying attention to the business cost of false positives and false negatives. Without labels, I would review a sample of high-ranked rows with a domain expert and use synthetic anomalies as a limited sanity check.

### Why use median imputation?
The median is less influenced by extreme values than the mean. It gives the model a complete numeric matrix, but I would also monitor missingness because missing values can carry business meaning.

### What is the anomaly score?
It is a relative ranking derived from the model's decision function. Higher values in this app mean more unusual according to the model; it is not a calibrated probability.

### Why Streamlit?
It allows a Python-focused prototype to become interactive quickly, making it easy for an analyst to upload data, adjust settings, view charts, and export results.

### Is this an LLM agent?
Not yet. This version uses machine learning for detection and deterministic logic for the report. I would add an LLM only if natural-language investigation summaries provide value, and I would ground it in computed results, protect sensitive data, validate outputs, and store API keys in secrets.

### What are the limitations?
Only selected numeric fields are modeled; the synthetic data is not proof of business impact; contamination requires a choice; anomalies may be legitimate; and the report is not an autonomous causal explanation.

## Demo sequence
1. Load the sample orders.
2. Explain the row count, missing cells, duplicates, and column quality table.
3. Choose `quantity`, `unit_price`, `revenue`, and `processing_minutes`.
4. Run detection and explain the flag rate.
5. Use the scatter plot to inspect the relationship between quantity and revenue.
6. Read the investigation brief.
7. Download the flagged records.
8. Close by explaining limitations and evaluation plans.

## Be ready to discuss the code
Do not claim a feature you have not tested. Run the app, test file upload and export, and be able to explain each module in your own words.

## Business purpose and email alert
Business teams can have thousands of transactions; aggregate dashboards may hide unusual individual records. This app screens a CSV and prioritizes a review queue for a data or operations analyst. After reviewing results, the analyst can click **Send anomaly alert email** to notify a configured recipient of the flagged count and top record IDs. The alert asks for investigation; it does not declare fraud.

The app is a CSV-based prototype, not a live monitoring service. It does not schedule daily scans. The report is rule-based, not LLM-generated. A production version could add scheduled ingestion, duplicate-alert suppression, audit logs, labelled model evaluation, and optionally grounded LLM summaries.

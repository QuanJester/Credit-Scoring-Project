# 🏦 Automated Credit Scoring Pipeline

An end-to-end automated machine learning pipeline built to ingest, preprocess, engineer features, and train models for Credit Risk Assessment. The pipeline is designed specifically to handle highly imbalanced credit datasets (approx. 8% default rate) and track all preprocessing configurations and model metrics via MLflow.

---

## 📊 Pipeline Workflow

Below is the workflow showing how raw data flows through custom Scikit-learn transformers to build the final model:

```mermaid
graph TD
    A[Raw CSV/JSON Data] --> B[DataLoader]
    B --> C[SchemaCreating]
    C -->|Generates data_schema.json| D[AutomaticEncoding]
    D --> E[AutomaticFeatureDrop]
    E --> F[AutomaticImputing]
    F --> G[FeatureEngineering]
    G --> H[AutomaticScaling]
    H --> I[Logistic Regression / LGBM Classifier]
    I --> J[Persist pipeline to baseline.pkl]
    I --> K[Log Parameters & Metrics to MLflow]

📁 Project Structure
├── config/
│   └── data_schema.json       # Inferred data types schema configuration
├── models/
│   └── baseline.pkl           # joblib binary of the trained pipeline
├── notebooks/                 # Jupyter Notebooks for EDA & research
├── raw_data/                  # Directory for raw CSV/JSON datasets (Ignored by Git)
├── src/
│   ├── data_ingestion.py      # Data loading components
│   ├── data_preprocessing.py  # Custom transformers (Encoding, Imputing, Dropping, Scaling)
│   ├── features.py            # Business logic features (Feature Engineering)
│   └── train.py               # Pipeline orchestration & MLflow tracking
├── .gitignore                 # Specifies files/directories to be ignored by Git
├── mlflow.db                  # Local SQLite database storing MLflow runs
├── requirements.txt           # Python library dependencies
└── test-code.py               # Simple test entrypoint script

🛠️ Detailed Component & Class Documentation
1. Data Ingestion (src/data_ingestion.py)
DataLoader
Purpose: Safely loads datasets in CSV and JSON formats with memory-safe defaults.
Internal Logic:
Employs pathlib.Path to verify file existence and checks file size (st_size == 0) to prevent loading empty files.
Supports flexible **kwargs parameters (such as nrows), allowing the system to read only a fraction of a massive dataset to build schema mappings without loading gigabytes of data into RAM.
2. Preprocessing & Custom Transformers (src/data_preprocessing.py)
MissingValueTracker
Purpose: Performs fast analysis of missing data ratio in each column and classifies the missing-risk level.
Internal Logic:
Uses vectorized Pandas computations (df.isnull().mean() * 100) to compute missing percentages efficiently.
Leverages binary search (bisect.bisect_left) against a risk threshold dictionary (Low: 5%, Considerable: 20%, High: 50%, Danger: 100%) to categorize risk levels for each feature.
SchemaCreating
Purpose: Analyzes raw data columns to dynamically infer data types and categorization rules, saving them to config/data_schema.json.
Internal Logic:
Reads a configurable number of preview rows (default: 1000).
Classifies features into numerical_column, one_hot_column, or ordinal_column.
Cardinality Rule: Categorical features with unique values ≤ cardinality_threshold (default: 5) are routed to One-Hot Encoding; features with higher cardinality are routed to Ordinal Encoding.
Automatically excludes ID and Target columns (SK_ID_CURR, TARGET) to prevent target leakage.
SchemaLoading
Purpose: Utility class that opens the config/data_schema.json file and returns a structured dictionary for downstream transformers.
AutomaticEncoding
Purpose: Custom Scikit-learn transformer that applies One-Hot Encoding and Ordinal Encoding dynamically based on the generated schema.
Internal Logic:
One-Hot Encoding: Instantiated with handle_unknown="ignore" and sparse_output=False. It automatically renames generated dummy columns (get_feature_names_out) and concatenates them back to the DataFrame while dropping the raw column.
Ordinal Encoding: Instantiated with handle_unknown="use_encoded_value" and unknown_value=-1 to prevent crashes when encountering unseen labels in the test or production datasets.
AutomaticFeatureDrop
Purpose: Dropping useless columns containing high missing values unless they possess high predictive power.
Internal Logic (Pipeline-Safe):
Feature Importance Assessment: Fits a fast LGBMClassifier (50 estimators) in the fit() method and extracts feature importances. To ensure scale-invariance, it normalizes importances: Normalized Importance= 
∑feature_importance_
feature_importance_
​
 
Drop Rule: In fit(), it calculates the missing value ratio. If a feature's missing ratio >50% (default) AND its normalized importance is below the importance_threshold, it is added to a local self.feature_dropped list.
No Data Leakage / Consistency: Inside transform(), it drops the pre-calculated columns in self.feature_dropped if they exist. No missing values or model importances are computed during transform(), ensuring the train and test shapes are always identical.
AutomaticImputing
Purpose: Automatically imputes missing values based on statistical distribution characteristics.
Internal Logic:
Categorical Columns: Replaces missing values with the Mode (most frequent category).
Numerical Columns: Computes the absolute skewness value (abs(df[col].skew())).
If skewness ≤0.5 (normally distributed data): Imputes using the Mean.
If skewness >0.5 (skewed data): Imputes using the Median to avoid bias caused by outliers.
Calculates values during fit() and applies them via .fillna() during transform().
AutomaticScaling
Purpose: Scales numerical features dynamically based on skewness and the presence of outliers.
Internal Logic:
Outlier Detection: Uses the Interquartile Range (IQR): IQR=Q3−Q1 Lower Bound=Q1−1.5×IQR,Upper Bound=Q3+1.5×IQR Values outside these bounds are marked as outliers.
Scaler Selection Rules:
If Outlier Percentage >1%: Fits RobustScaler (uses median and quantiles, making it resistant to outliers).
If Outlier Percentage ≤1% AND Skewness ≤0.5: Fits StandardScaler (best for normally distributed features).
If Outlier Percentage ≤1% AND Skewness >0.5: Fits MinMaxScaler (scales bounded values within [0,1] without outlier interference).
3. Feature Engineering (src/features.py)
FeatureEngineering
Purpose: Stretches domain-specific feature engineering rules for credit scoring models.
Internal Logic:
Creates credit_income_ratio representing total income divided by the credit amount: Credit-Income Ratio= 
AMT_CREDIT
AMT_INCOME_TOTAL
​
 
Safely checks for the existence of columns before applying math operations to avoid crashes if prior steps dropped the columns.
4. Training Pipeline (src/train.py)
TrainingPipeline
Purpose: Bundles data loading, splitting, preprocessing steps, engineering, model training, and MLflow logging into a unified pipeline.
Internal Logic:
Splits raw data into train/test subsets (test_size=0.2).
Stacks the custom transformers into an sklearn.pipeline.Pipeline.
Class Imbalance Solution: Employs class_weight="balanced" in the estimator (e.g., LogisticRegression) to adjust loss penalties inversely proportional to class frequencies, countering the severe class imbalance (~8% defaults).
Logging (MLflow): Connects to a local SQLite database (sqlite:///mlflow.db). Logs metrics (Precision, Recall, F1, and ROC-AUC) along with hyperparameters (skewness, outlier percentage, and classification parameters) to the tracking dashboard.
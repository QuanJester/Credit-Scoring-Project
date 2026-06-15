import pandas as pd
from pathlib import Path
import numpy as np
import bisect
import json
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from data_ingestion import DataLoader
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import matplotlib.pyplot as plt
# class to check missing ratio and classify missing risk in each column (feature)
class MissingValueTracker:
    def __init__(self, data: pd.DataFrame, missing_risk_threshold: dict = None):
        self.data = data
        self.missing_risk_threshold = missing_risk_threshold if missing_risk_threshold is not None else {
            "Low": 5.0, 
            "Considerable": 20.0, 
            "High": 50.0,
            "Danger": 100.0
        }

    def calculate_missing_ratio(self) -> pd.Series:
        df = self.data
        # Vectorized calculation of missing ratios (extremely fast)
        missing_ratio_series = (df.isnull().mean() * 100).round(4)
        
        sorted_thresholds = sorted(self.missing_risk_threshold.items(), key=lambda item: item[1])
        risk_categories, risk_thresholds = zip(*sorted_thresholds)
        
        missing_risk_category = {}
        for column, ratio in missing_ratio_series.items():
            category_idx = bisect.bisect_left(risk_thresholds, ratio)
            category_idx = min(category_idx, len(risk_thresholds) - 1)
            missing_risk_category[column] = risk_categories[category_idx]
            
        return pd.Series(missing_risk_category)
# detect outliers
class Outliers():
    def __init__(self, df: pd.DataFrame):
        self.df = df
    def plot_and_detect_outliers(self, column):
        data = self.df
        q1 = data[column].quantile(0.25)
        q3 = data[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        total_outliers = (data[column] < lower_bound or data[column] > upper_bound)
        print(f"Total outliers detected: {total_outliers}")
        plt.figure(figsize=(10, 6))
        # Vẽ giá trị DAYS_EMPLOYED theo số thứ tự dòng (index)
        plt.scatter(
            x=data.index, 
            y=data[column], 
            alpha=0.5, 
            c='green', 
            s=10 # kích thước điểm vẽ
        )

        plt.title(f"Scatter plot of {column}")
        plt.xlabel('Index')
        plt.ylabel(f"{column}")
        plt.grid(True)
        plt.show()
# class to automatically impute missing value based on the feature's distribution
class AutomaticImputing(BaseEstimator, TransformerMixin):
    def __init__(self, skew_threshold = 0.5):
        self.skew_threshold = skew_threshold
        self.impute_values_ = {}

    def fit(self, x, y=None):
        if not isinstance(x, pd.DataFrame):
            raise ValueError("X must be a dataframe")
        self.impute_values_ = {}
        
        # Separate numerical and categorical columns
        numerical_columns = x.select_dtypes(include=[np.number]).columns
        categorical_columns = x.select_dtypes(exclude=[np.number]).columns
        
        for feature in numerical_columns:
            feature_skewness = abs(float(np.round(x[feature].skew(), 4)))
            if feature_skewness <= self.skew_threshold:
                self.impute_values_[feature] = float(np.round(x[feature].mean(), 4))
            else:
                self.impute_values_[feature] = float(np.round(x[feature].median(), 4))
                
        for feature in categorical_columns:
            non_null_data = x[feature].dropna()
            if not non_null_data.empty:
                self.impute_values_[feature] = non_null_data.mode()[0]
                
        return self

    def transform(self, x):
        if not isinstance(x, pd.DataFrame):
            raise ValueError("X must be a dataframe")
        
        x = x.copy()  # Avoid side effects on the original dataframe
        for feature, impute_value in self.impute_values_.items():
            if feature in x.columns:
                x[feature] = x[feature].fillna(impute_value)
        print("Feature imputing completed.")
        return x

# class to automatically encoding categorical feature
class AutomaticEncoding(BaseEstimator, TransformerMixin):
    def __init__(self, schema_filepath: str):
        self.schema_filepath = schema_filepath
        self.ordinal_encode_ = {}
        self.onehot_encode_ = {}
        self.onehot_cols = []
        self.ordinal_cols = []
    def load_from_schema(self) -> dict:
        with open(self.schema_filepath, "r", encoding = 'utf-8') as f:
            schema_file = json.load(f)
        return dict(schema_file)
    def fit(self, x, y=None):
        self.onehot_cols = []
        self.ordinal_cols = []
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(x)
        x = x.copy()
        category_dictionary = self.load_from_schema()
        #load columns from schema file to 2 seperate columns
        for key in category_dictionary.keys():
            if key == "one_hot_column":
                self.onehot_cols.extend(category_dictionary[key])
            if key == "ordinal_column":
                self.ordinal_cols.extend(category_dictionary[key])
        for column in self.onehot_cols:
            if column in x.columns:
                encoder = OneHotEncoder(handle_unknown= "ignore", sparse_output= False)
                encoder.fit(x[[column]])
                self.onehot_encode_[column] = encoder
        for column in self.ordinal_cols:
            if column in x.columns:
                encoder = OrdinalEncoder(handle_unknown= "use_encoded_value", unknown_value= -1)
                encoder.fit(x[[column]])
                self.ordinal_encode_[column] = encoder
        return self
    def transform(self,x):
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(x)
        x = x.copy()
        for column, encoder in self.onehot_encode_.items():
            if column in x.columns:
                encoded_data = encoder.transform(x[[column]])
                new_col = encoder.get_feature_names_out([column])
                new_df = pd.DataFrame(encoded_data, columns= new_col, index = x.index)

                x = x.drop(columns=[column])
                x = pd.concat([x, new_df], axis = 1)
        for column, encoder in self.ordinal_encode_.items():
            if column in x.columns:
                x[column] = encoder.transform(x[[column]])
        print(f"Feature encoding completed: data shape: {x.shape}")
        return x
        

# class to create a schema file contains metadata of each column
class SchemaCreating():
    def __init__(self, cardinality_threshold: int = 5, target_column: str = "TARGET", id_column: str = "SK_ID_CURR"):
        self.cardinality_threshold = cardinality_threshold
        self.target_column = target_column
        self.id_column = id_column
        self.schema_ = {}

    def fit_from_csv(self, file_path: str, nrows: int = 1000) -> dict:
        # Pass nrows to avoid loading massive CSV files fully
        data = DataLoader.load_csv(file_path, nrows=nrows)
        self.schema_ = {
            "target_column": self.target_column, 
            "id_column": self.id_column,
            "numerical_column": [],
            "one_hot_column": [],
            "ordinal_column": []
        }
        
        for column in data.columns:
            # Exclude ID and target columns from feature lists
            if column in (self.target_column, self.id_column):
                continue
                
            if pd.api.types.is_numeric_dtype(data[column]):
                self.schema_["numerical_column"].append(column)
            else:
                column_unique = data[column].nunique()
                if column_unique <= self.cardinality_threshold:
                    self.schema_["one_hot_column"].append(column)
                else:
                    self.schema_["ordinal_column"].append(column)
                    
        print(f"File is successfully classified")
        return self.schema_

    def save_to_json(self, output_path: str):
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.schema_, f, indent=4, ensure_ascii=False)

# class to automatically drop columns that contain too many missing values
class AutomaticFeatureDrop(BaseEstimator, TransformerMixin):
    def __init__(self, importance_threshold: float = 0.65, missing_ratio_threshold: float = 50.0):
        self.importance_threshold = importance_threshold
        self.missing_ratio_threshold = missing_ratio_threshold
        self.feature_importance_ = {} #contains feature importance of all features
        self.important_feature_ = {} # contains feature importance >= threshold only feature
        self.feature_dropped = []
    def filt_important_feature(self, x: pd.DataFrame, y: pd.Series):
        #using lightGBM to get feature_importance
        model = LGBMClassifier(n_estimators=50, random_state= 42, verbose = -1)
        model.fit(x,y)
        feature_importance = model.feature_importances_
        feature_name = x.columns
        for i in range(0, len(feature_name)):
            self.feature_importance_[feature_name[i]] = feature_importance[i] / feature_importance.sum()
    def fit(self, x, y: None):
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(x)
        x = x.copy()
        self.filt_important_feature(x, y)
        for feature, feature_importance in self.feature_importance_.items():
            if feature_importance > self.importance_threshold:
                self.important_feature_[feature] = feature_importance
        for column in x.columns:
            if (x[column].isnull().sum() == 0) | (column in self.important_feature_.keys()):
                continue
            total_row = len(x)
            missing_ratio = float(np.round(x[column].isnull().sum() / total_row, 4)) * 100
            if missing_ratio > self.missing_ratio_threshold:
                self.feature_dropped.append(column)
        return self
    def transform(self, x):
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(x)
        x = x.copy()
        for col in self.feature_dropped:
            if col in x.columns:
                x = x.drop(col, axis = 1)
        print(f"Feature dropping completed, data shape; {x.shape}, feature dropped: {len(self.feature_dropped)}")
        return x

# class to automatically scale features based on distribution or outliers
class AutomaticScaling(BaseEstimator, TransformerMixin):
    def __init__(self, outlier_percentage: int = 1, schema_file_path: str = "./config/data_schema.json"):
        self.feature_scaling_ = {}
        self.skew_threshold = 0.5
        self.outlier_percentage = outlier_percentage
        self.schema_file_path = schema_file_path
    def fit(self, x ,y=None):
        self.feature_scaling_ = {}
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(x)
        x = x.copy()
        feature_category = SchemaLoading(self.schema_file_path).load_schema()
        numerical_feature = feature_category['numerical_column'].copy()
        # Add engineered features if they are in the dataset
        for col in ['credit_income_ratio', 'income_annuity_ratio', 'birth_employed_ratio', 'credit_goods_ratio']:
            if col in x.columns and col not in numerical_feature:
                numerical_feature.append(col)
        for column in numerical_feature:
            if column not in x.columns:
                continue
            skew = float(np.round(x[column].skew(), 4))
            q1 = x[column].quantile(0.25)
            q3 = x[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            outliers = ((x[column] < lower_bound) | (x[column] > upper_bound)).sum()
            total_row = len(x)
            outlier_percentage = float(np.round(outliers / total_row, 4)) * 100
            if outlier_percentage > self.outlier_percentage:
                scaler = RobustScaler(quantile_range= (25,75))
                scaler.fit(x[[column]])
                self.feature_scaling_[column] = scaler
            elif skew <= self.skew_threshold: # applying standard scaler for Normal distribution feature
                scaler = StandardScaler()
                scaler.fit(x[[column]])
                self.feature_scaling_[column] = scaler
            else:
                scaler = MinMaxScaler()
                scaler.fit(x[[column]])
                self.feature_scaling_[column] = scaler
        return self
    def transform(self, x):
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(x)
        x = x.copy()
        for column, scaler in self.feature_scaling_.items():
            if column in x.columns:
                x[column] = scaler.transform(x[[column]])
        print("Data scaled completed")
        return x
            
#class to load schema file
class SchemaLoading:
    def __init__(self, file_path: str):
        self.file_path = file_path
    def load_schema(self) -> dict:
        with open(self.file_path, "r", encoding = "utf-8") as f:
            schema_file = json.load(f)
        return dict(schema_file)
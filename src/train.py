from data_ingestion import DataLoader
from data_preprocessing import AutomaticEncoding, AutomaticFeatureDrop, AutomaticImputing, AutomaticScaling
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from features import FeatureEngineering
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from lightgbm import LGBMClassifier
import pandas as pd
import joblib
import mlflow
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class TrainingPipeline:
    def __init__(self, file_path: str, schema_path: str , model_save_path: str ):
        self.file_path = file_path
        self.schema_path = schema_path
        self.pipeline = None 
        self.model_save_path = model_save_path
    def load_and_split(self, test_size: float = 0.2):
        data = DataLoader.load_csv(self.file_path)
        
        # Merge auxiliary datasets dynamically using parent folder of file_path
        raw_data_dir = str(Path(self.file_path).parent)
        data = DataLoader.merge_auxiliary_data(data, raw_data_dir)
        
        x = data.drop('TARGET', axis = 1).copy()
        y = data['TARGET'].copy()
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=test_size, random_state= 42, stratify=y)
        return x_train, x_test, y_train, y_test
    #function to create training pipeline
    def create_pipeline(self):
        self.pipeline = Pipeline([
            ('Encoding', AutomaticEncoding(schema_filepath= self.schema_path)),
            ('Feature Dropping', AutomaticFeatureDrop(importance_threshold= 0.01)),
            ('Feature engineering', FeatureEngineering()),
            ('Imputing', AutomaticImputing()),
            ('Scaling', AutomaticScaling()), 
            ("Model training", LGBMClassifier(random_state= 42, n_estimators= 250, learning_rate= 0.03, class_weight= "balanced", verbose = -1))
        ])
    #function to train model on a whole data set
    def training(self):
        print("Initializing training...")
        x_train, x_test, y_train, y_test = self.load_and_split()
        self.create_pipeline()
        print(f"Fitting training set: \n x_train: {x_train.shape} \n y_train: {y_train.shape}")
        self.pipeline.fit(x_train, y_train)
        print(f"Predict test set: {x_test.shape}")
        prediction = self.pipeline.predict(x_test)  
        predict_proba = self.pipeline.predict_proba(x_test)[:, 1]
        precision, recall, f1, roc_auc = precision_score(y_test, prediction), recall_score(y_test, prediction), f1_score(y_test, prediction), roc_auc_score(y_test, predict_proba)
        print(f"Precision score: {precision} \nRecall score: {recall} \nF1 score: {f1} \nRoc auc: {roc_auc}")
        self.save_pipeline()
        #tracking training with ml flow
        mlflow.set_tracking_uri("sqlite:///mlflow.db") 
        mlflow.set_experiment("Credit Scoring")
        with mlflow.start_run(run_name="Test 13: create new features (payment_rate, ext_std, high_default_risk) and merge datasets"):
            # baseline parameters from preprocessing pipeline
            mlflow.log_param("importance threshold", self.pipeline.named_steps['Feature Dropping'].importance_threshold)
            mlflow.log_param("skew threshold", self.pipeline.named_steps['Imputing'].skew_threshold)
            mlflow.log_param("outlier percentage threshold" , self.pipeline.named_steps['Scaling'].outlier_percentage)
            mlflow.log_param("n estimators", self.pipeline.named_steps['Model training'].n_estimators)
            mlflow.log_param("class weight", self.pipeline.named_steps['Model training'].class_weight)
            mlflow.log_param("learning rate", self.pipeline.named_steps['Model training'].learning_rate)
            # metrics score
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1", f1)
            mlflow.log_metric("roc auc", roc_auc)
    #function to cross_validate  
    def cross_validate(self, n_splits: int = 5):
        data = DataLoader.load_csv(self.file_path)
        
        # Merge auxiliary datasets dynamically
        raw_data_dir = str(Path(self.file_path).parent)
        data = DataLoader.merge_auxiliary_data(data, raw_data_dir)
        
        x = data.drop('TARGET', axis = 1).copy()
        y = data['TARGET'].copy()
        precision_list = []
        recall_list = []
        f1_list = []
        roc_list = []
        stf = StratifiedKFold(n_splits= n_splits, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(stf.split(x,y)):
            x_train_fold, x_val_fold = x.iloc[train_idx], x.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            self.create_pipeline()
            self.pipeline.fit(x_train_fold, y_train_fold)
            val_preds = self.pipeline.predict(x_val_fold)
            val_proba = self.pipeline.predict_proba(x_val_fold)[:, 1]
            precision = precision_score(y_val_fold, val_preds)
            recall = recall_score(y_val_fold, val_preds)
            f1 = f1_score(y_val_fold, val_preds)
            roc = roc_auc_score(y_val_fold, val_proba)
            print(f"Precision: {precision}, Recall: {recall}, F1: {f1}, ROC-AUC: {roc}")
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            roc_list.append(roc)
        precision_mean = np.mean(precision_list)
        recall_mean = np.mean(recall_list)
        f1_mean = np.mean(f1_list)
        roc_mean = np.mean(roc_list)
        mlflow.set_tracking_uri("sqlite:///mlflow.db") 
        mlflow.set_experiment("Credit Scoring")
        with mlflow.start_run(run_name="Test 2 (Cross validate with merged data)"):
             # parameters
             mlflow.log_param("importance threshold", self.pipeline.named_steps['Feature Dropping'].importance_threshold)
             mlflow.log_param("skew threshold", self.pipeline.named_steps['Imputing'].skew_threshold)
             mlflow.log_param("outlier percentage threshold", self.pipeline.named_steps['Scaling'].outlier_percentage)
             mlflow.log_param("n estimators", self.pipeline.named_steps['Model training'].n_estimators)
             mlflow.log_param("n_splits", n_splits)
             mlflow.log_param("class weight", self.pipeline.named_steps['Model training'].class_weight)
             mlflow.log_param("learning rate", self.pipeline.named_steps['Model training'].learning_rate)
             # metrics score
             mlflow.log_metric("mean precision", precision_mean)
             mlflow.log_metric("mean recall", recall_mean)
             mlflow.log_metric("mean f1", f1_mean)
             mlflow.log_metric("mean roc auc", roc_mean)
    #function to tune hyperparameters
    def full_pipelinehyperparameter_tuning(self):
        x_train, x_test, y_train, y_test = self.load_and_split()
        # tuning hyperparameter of preprocessing pipeline
        param_dist = {
            # Corrected tuning search space to prevent dropping important features
            'Feature Dropping__importance_threshold': [0.005, 0.01, 0.015, 0.02],
            'Imputing__skew_threshold': [0.4, 0.5],
            'Scaling__outlier_percentage': [1,2,3,4,5],
            'Model training__n_estimators': [100,150,200,250],
            'Model training__learning_rate': [0.03,0.05,0.07],
            'Model training__class_weight': ['balanced']
        }
        self.create_pipeline()
        random_search = RandomizedSearchCV(
            estimator= self.pipeline, 
            param_distributions= param_dist, 
            n_iter= 5, 
            cv = 3, 
            scoring= 'roc_auc',
            n_jobs= -1, 
            verbose=1
        )
        print(f"Tuning hyperparameter...")
        random_search.fit(x_train, y_train)
        best_param = random_search.best_params_
        print(f"Best params: {best_param}")
        print(f"Best score: {random_search.best_score_:.4f}")
        self.pipeline = random_search.best_estimator_
        self.save_pipeline()
        #testing 
        prediction = random_search.predict(x_test)
        predict_proba = random_search.predict_proba(x_test)[:, 1]
        precision = precision_score(y_test, prediction)
        recall = recall_score(y_test, prediction)
        f1 = f1_score(y_test, prediction)
        roc_auc = roc_auc_score(y_test, predict_proba)
        mlflow.set_tracking_uri("sqlite:///mlflow.db") 
        mlflow.set_experiment("Credit Scoring")
        with mlflow.start_run(run_name="Test 3 (Hyperparameter tuning)"):
            for param, value in best_param.items():
                mlflow.log_param(param, value)
            # metrics score
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1", f1)
            mlflow.log_metric("roc auc", roc_auc)
        
    def save_pipeline(self):
        joblib.dump(self.pipeline, self.model_save_path)
        print(f"Model saved at {self.model_save_path}")


         


if __name__ == "__main__":
    trainer = TrainingPipeline(
        file_path= "./raw_data/data_merged.csv",
        schema_path = "./config/data_schema_full.json",
        model_save_path= "./models/full.pkl"
    )
    trainer.training()
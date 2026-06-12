from data_ingestion import DataLoader
from data_preprocessing import AutomaticEncoding, AutomaticFeatureDrop, AutomaticImputing, AutomaticScaling
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from features import FeatureEngineering
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import joblib
import mlflow
class TrainingPipeline:
    def __init__(self, file_path: str, schema_path: str, model_save_path: str):
        self.file_path = file_path
        self.schema_path = schema_path
        self.pipeline = None 
        self.model_save_path = model_save_path
    def load_and_split(self, test_size: float = 0.2):
        data = DataLoader.load_csv(self.file_path)
        x = data.drop('TARGET', axis = 1).copy()
        y = data['TARGET'].copy()
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=test_size, random_state= 42)
        return x_train, x_test, y_train, y_test
    def create_pipeline(self):
        self.pipeline = Pipeline([
            ('Encoding', AutomaticEncoding(schema_filepath= self.schema_path)),
            ('Feature Dropping', AutomaticFeatureDrop(importance_threshold= 0.8)),
            ('Imputing', AutomaticImputing()),
            ('Feature engineering', FeatureEngineering()),
            ('Scaling', AutomaticScaling()), 
            ("Model training", LogisticRegression(random_state=42, max_iter= 1500,class_weight="balanced"))
        ])
    def training(self):
        print("Initializing training...")
        x_train, x_test, y_train, y_test = self.load_and_split()
        self.create_pipeline()
        print(f"Fitting training set: \n x_train: {x_train.shape} \n y_train: {y_train.shape}")
        self.pipeline.fit(x_train, y_train)
        print(f"Predict test set: {x_test.shape}")
        prediction = self.pipeline.predict(x_test)  
        predict_proba = self.pipeline.predict_proba(x_test)[:, 1]
        precision, recall, f1, roc_auc = precision_score(y_test, prediction), recall_score(y_test, prediction), f1_score(y_test, prediction), roc_auc_score(y_test, prediction)
        print(f"Precision score: {precision} \nRecall score: {recall} \nF1 score: {f1} \nRoc auc: {roc_auc}")
        self.save_pipeline()
        #tracking training with ml flow
        mlflow.set_tracking_uri("sqlite:///mlflow.db") 
        mlflow.set_experiment("Credit Scoring")
        with mlflow.start_run(run_name="Test 1: creating new feature (income/credit) - Logistic Regression"):
            # baseline parameters from preprocessing pipeline
            mlflow.log_param("importance threshold", self.pipeline.named_steps['Feature Dropping'].importance_threshold)
            mlflow.log_param("skew threshold", self.pipeline.named_steps['Imputing'].skew_threshold)
            mlflow.log_param("outlier percentage threshold", self.pipeline.named_steps['Scaling'].outlier_percentage)
            mlflow.log_param("max iterations", self.pipeline.named_steps['Model training'].max_iter)
            mlflow.log_param("class weight", self.pipeline.named_steps['Model training'].class_weight)
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
        file_path= "./raw_data/application_train.csv",
        schema_path = "./config/data_schema.json",
        model_save_path= "./models/baseline.pkl"
    )
    trainer.training()
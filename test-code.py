file_path = './raw_data/application_train.csv'
schema_path = "./config/data_schema.json"
model_save_path = "./models/baseline.pkl"
from src.train import TrainingPipeline
model = TrainingPipeline(file_path, schema_path, model_save_path)
model.training()
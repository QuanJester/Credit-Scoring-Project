from src.data_ingestion import DataLoader
file_path = "./raw_data/application_train.csv"
data = DataLoader.load_csv(file_path=file_path)
column = 'DAYS_BIRTH'
from src.data_preprocessing import Outliers
Outliers(data).plot_and_detect_outliers(column)

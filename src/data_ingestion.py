import pandas as pd
import logging
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
# loading data from csv and json files
class DataLoader:
    @staticmethod
    def load_csv(file_path: str, **kwargs) -> pd.DataFrame:
        path = Path(file_path)
        if not path.exists():
            raise ValueError(f"File path {path} does not exist.")
        if path.stat().st_size == 0:
            raise ValueError(f"File {path.name} is empty.")
        try: 
            data = pd.read_csv(path, **kwargs)
            print(f"File {path.name} is loaded successfully, data shape: {data.shape}")
        except Exception as unknown_error:
            raise ValueError(f"File {path.name} exists but some error prevented loading: {str(unknown_error)}") from unknown_error
        return data
        
    @staticmethod
    def load_json(file_path: str) -> dict:
        """Loads and returns dictionary data from a JSON configuration/schema file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"JSON file {path} does not exist.")
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"File {path.name} loaded successfully.")
            return data
        except Exception as e:
            raise ValueError(f"Error reading JSON file {path.name}: {str(e)}") from e

    @staticmethod
    def load_json_to_df(file_path: str, **kwargs) -> pd.DataFrame:
        """Loads a JSON dataset file and returns a Pandas DataFrame."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"JSON file {path} does not exist.")
        try:
            data = pd.read_json(path, **kwargs)
            print(f"JSON dataset {path.name} loaded successfully as DataFrame.")
            return data
        except Exception as e:
            raise ValueError(f"Error reading JSON dataset {path.name} as DataFrame: {str(e)}") from e
    

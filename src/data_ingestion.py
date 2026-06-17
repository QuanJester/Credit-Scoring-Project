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
            
    @staticmethod
    def merge_auxiliary_data(main_df: pd.DataFrame, raw_data_dir: str) -> pd.DataFrame:
        """
        Reads, aggregates, and merges auxiliary datasets (bureau, previous_application,
        installments_payments, POS_CASH_balance, credit_card_balance) into the main dataframe.
        """
        import gc
        raw_dir = Path(raw_data_dir)
        df = main_df.copy()
        
        # 1. Merge bureau.csv
        bureau_path = raw_dir / "bureau.csv"
        if bureau_path.exists():
            print("Processing and merging bureau.csv...")
            try:
                bureau = pd.read_csv(bureau_path)
                bureau_numeric_agg = bureau.groupby('SK_ID_CURR').agg({
                    'DAYS_CREDIT': ['min', 'max', 'mean'],
                    'CREDIT_DAY_OVERDUE': ['max', 'mean'],
                    'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
                    'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
                    'CNT_CREDIT_PROLONG': ['sum']
                })
                bureau_numeric_agg.columns = pd.Index([f"BUREAU_{e[0]}_{e[1].upper()}" for e in bureau_numeric_agg.columns.tolist()])
                bureau_numeric_agg = bureau_numeric_agg.reset_index()
                df = df.merge(bureau_numeric_agg, on='SK_ID_CURR', how='left')
                del bureau, bureau_numeric_agg
                gc.collect()
            except Exception as e:
                print(f"Warning: Failed to merge bureau.csv: {e}")
        
        # 2. Merge previous_application.csv
        prev_path = raw_dir / "previous_application.csv"
        if prev_path.exists():
            print("Processing and merging previous_application.csv...")
            try:
                prev = pd.read_csv(prev_path)
                prev_numeric_agg = prev.groupby('SK_ID_CURR').agg({
                    'AMT_ANNUITY': ['max', 'mean'],
                    'AMT_APPLICATION': ['max', 'mean'],
                    'AMT_CREDIT': ['max', 'mean'],
                    'AMT_DOWN_PAYMENT': ['max', 'mean'],
                    'DAYS_DECISION': ['min', 'max', 'mean'],
                    'CNT_PAYMENT': ['mean', 'sum']
                })
                prev_numeric_agg.columns = pd.Index([f"PREV_{e[0]}_{e[1].upper()}" for e in prev_numeric_agg.columns.tolist()])
                prev_numeric_agg = prev_numeric_agg.reset_index()
                df = df.merge(prev_numeric_agg, on='SK_ID_CURR', how='left')
                del prev, prev_numeric_agg
                gc.collect()
            except Exception as e:
                print(f"Warning: Failed to merge previous_application.csv: {e}")
                
        # 3. Merge installments_payments.csv
        inst_path = raw_dir / "installments_payments.csv"
        if inst_path.exists():
            print("Processing and merging installments_payments.csv...")
            try:
                inst = pd.read_csv(inst_path)
                inst['PAYMENT_DELAY'] = inst['DAYS_ENTRY_PAYMENT'] - inst['DAYS_INSTALMENT']
                inst['PAYMENT_DELTA'] = inst['AMT_INSTALMENT'] - inst['AMT_PAYMENT']
                inst_agg = inst.groupby('SK_ID_CURR').agg({
                    'NUM_INSTALMENT_VERSION': ['max'],
                    'DAYS_INSTALMENT': ['max', 'mean'],
                    'DAYS_ENTRY_PAYMENT': ['max', 'mean'],
                    'AMT_INSTALMENT': ['mean', 'sum'],
                    'AMT_PAYMENT': ['mean', 'sum'],
                    'PAYMENT_DELAY': ['max', 'mean'],
                    'PAYMENT_DELTA': ['max', 'mean', 'sum']
                })
                inst_agg.columns = pd.Index([f"INST_{e[0]}_{e[1].upper()}" for e in inst_agg.columns.tolist()])
                inst_agg = inst_agg.reset_index()
                df = df.merge(inst_agg, on='SK_ID_CURR', how='left')
                del inst, inst_agg
                gc.collect()
            except Exception as e:
                print(f"Warning: Failed to merge installments_payments.csv: {e}")

        # 4. Merge POS_CASH_balance.csv
        pos_path = raw_dir / "POS_CASH_balance.csv"
        if pos_path.exists():
            print("Processing and merging POS_CASH_balance.csv...")
            try:
                pos = pd.read_csv(pos_path)
                pos_agg = pos.groupby('SK_ID_CURR').agg({
                    'MONTHS_BALANCE': ['max', 'mean'],
                    'CNT_INSTALMENT': ['mean'],
                    'CNT_INSTALMENT_FUTURE': ['mean'],
                    'SK_DPD': ['max', 'mean'],
                    'SK_DPD_DEF': ['max', 'mean']
                })
                pos_agg.columns = pd.Index([f"POS_{e[0]}_{e[1].upper()}" for e in pos_agg.columns.tolist()])
                pos_agg = pos_agg.reset_index()
                df = df.merge(pos_agg, on='SK_ID_CURR', how='left')
                del pos, pos_agg
                gc.collect()
            except Exception as e:
                print(f"Warning: Failed to merge POS_CASH_balance.csv: {e}")

        # 5. Merge credit_card_balance.csv
        cc_path = raw_dir / "credit_card_balance.csv"
        if cc_path.exists():
            print("Processing and merging credit_card_balance.csv...")
            try:
                cc = pd.read_csv(cc_path)
                cc_agg = cc.groupby('SK_ID_CURR').agg({
                    'AMT_BALANCE': ['max', 'mean'],
                    'AMT_CREDIT_LIMIT_ACTUAL': ['max', 'mean'],
                    'AMT_DRAWINGS_CURRENT': ['max', 'mean', 'sum'],
                    'AMT_PAYMENT_CURRENT': ['mean', 'sum'],
                    'SK_DPD': ['max', 'mean']
                })
                cc_agg.columns = pd.Index([f"CC_{e[0]}_{e[1].upper()}" for e in cc_agg.columns.tolist()])
                cc_agg = cc_agg.reset_index()
                df = df.merge(cc_agg, on='SK_ID_CURR', how='left')
                del cc, cc_agg
                gc.collect()
            except Exception as e:
                print(f"Warning: Failed to merge credit_card_balance.csv: {e}")

        print(f"Data merge completed, final shape: {df.shape}")
        return df

    

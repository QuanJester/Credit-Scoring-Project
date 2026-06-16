import numpy as np
from typing import Self
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
#class to create new features
class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, x, y=None):
        return self
    def transform(self, x):
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame(x)
        x = x.copy()
        # calculating credit/income ratio
        if 'AMT_INCOME_TOTAL' in x.columns and 'AMT_CREDIT' in x.columns:
            x['credit_income_ratio'] = x['AMT_INCOME_TOTAL']/x['AMT_CREDIT']
        # annuity / income
        if 'AMT_INCOME_TOTAL' in x.columns and 'AMT_ANNUITY' in x.columns:
            x['income_annuity_ratio'] = x['AMT_INCOME_TOTAL'] / x['AMT_ANNUITY']
        # employment-to-age ratio
        if 'DAYS_BIRTH' in x.columns and 'DAYS_EMPLOYED' in x.columns:
            x['birth_employed_ratio'] = x['DAYS_BIRTH'] / x['DAYS_EMPLOYED']
        # credit-to-goods ratio
        if 'AMT_GOODS_PRICE' in x.columns and 'AMT_CREDIT' in x.columns:
            x['credit_goods_ratio'] = x['AMT_CREDIT'] / x['AMT_GOODS_PRICE']
        #mean exit source
        ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
        x['EXT_mean'] = x[ext_cols].mean(skipna=True, axis= 1)
        x['high_default_risk'] = np.where(x['EXT_mean'].isnull(), np.nan, (x['EXT_mean'] <= 0.439).astype(float))
        new_cols = ['credit_income_ratio', 'income_annuity_ratio', 'birth_employed_ratio', 'credit_goods_ratio', 'EXT_mean', 'high_default_risk']
        for cols in new_cols:
            x[cols] = x[cols].replace([np.inf, -np.inf], np.nan)
        print(f"New features created: {new_cols}")
        return x
        
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
        if 'AMT_INCOME_TOTAL' in x.columns and 'AMT_CREDIT' in x.columns:
            x['credit_income_ratio'] = x['AMT_INCOME_TOTAL']/x['AMT_CREDIT']
        return x
        
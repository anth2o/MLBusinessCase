from sklearn.base import BaseEstimator,TransformerMixin
import numpy as np
import pandas as pd

class Preprocessing(BaseEstimator, TransformerMixin):
    def __init__(self, is_store=True):
        if is_store:
            self.COLUMNS_DUMMY = [
                'PromoInterval',
                'StoreType',
                'Assortment'
            ]
            self.COLUMNS_CYCLIC = []
            self.COLUMNS_TEMPORAL = {
                'CompetitionOpenSince': [
                    ('CompetitionOpenSinceYear', 12),
                    ('CompetitionOpenSinceMonth', 1)
                ],
                'Promo2Since': [
                    ('Promo2SinceYear', 365.25 / 7),
                    ('Promo2SinceWeek', 1)
                ]
            }
        else:
            self.COLUMNS_DUMMY = []
            self.COLUMNS_CYCLIC = [
                'DayOfWeek'
            ]
            self.COLUMNS_TEMPORAL = {}
    # Mandatory method
    def fit(self, X_df, y):
        return self

    # Mandatory method
    def transform(self, X_df):
        X_df = self.fill_na(X_df)
        X_df = pd.get_dummies(X_df, columns=self.COLUMNS_DUMMY, drop_first=True)
        X_df = self.encode_cyclic_values(X_df)
        X_df = self.encode_temporal_values(X_df)
        return X_df.astype(np.float).fillna(0)

    def fill_na(self, X_df):
        X_df = X_df.fillna(X_df.median())
        X_df = X_df.fillna('')
        return X_df
    
    # We encode cyclic values by using trigonometric functions
    def encode_cyclic_values(self, X_df):
        for column in self.COLUMNS_CYCLIC:
            X_df['cos_{}'.format(column)] = np.cos(X_df[column] * 2 * np.pi / X_df[column].max())
            X_df['sin_{}'.format(column)] = np.sin(X_df[column] * 2 * np.pi / X_df[column].max())
        return X_df

    def encode_temporal_values(self, X_df):
        for key, value in self.COLUMNS_TEMPORAL.items():
            X_df[key] = pd.Series(np.zeros((X_df.shape[0],)), index=X_df.index)
            for i in X_df.index:
                X_df.loc[i, key] = (X_df.loc[i, value[0][0]] - 2000) * value[0][1] + X_df.loc[i, value[0][0]] * value[0][1]
            for keys_to_drop in value:
                X_df.drop(keys_to_drop[0], inplace=True, axis=1)
        return X_df

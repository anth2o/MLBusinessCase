from sklearn.base import BaseEstimator,TransformerMixin
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale



class Preprocessing(BaseEstimator, TransformerMixin):
    def fit(self, X_df, y):
        return self

    def transform(self, df_store, df_train):
        """
        This function takes as input two pandas dataframes:
            - df_store: a dataframe concerning stores (pd.read_csv('../data/store.csv'))
            - df_train: a dataframe concerning sales each day (pd.read_csv('../data/train.csv'))
        Ir returns:
            - A dataframe with features containing :
                - an embedding of each store done by PCA after having scaled the features of df_store.
                - df_train features after preprocessing
            - The sales for each row (the target)
        """
        df_store_preprocessed = self.transform_one_df(df_store, is_store=True)
        df_store_preprocessed = self.pca_df_store(df_store_preprocessed)
        df_train_preprocessed = self.transform_one_df(df_train, is_store=False)
        df_join = df_train_preprocessed.merge(df_store_preprocessed, left_on='Store', right_on='Store')
        return df_join.drop(['Sales', 'Storecd ..'], axis=1), df_join['Sales']

    def pca_df_store(self, df_store, n_components = 3) :
        df_store_bis = df_store.copy()
        df_store_bis = df_store_bis.drop(columns = ['Store'])
        df_store_bis = scale(df_store_bis)
        pca = PCA(n_components=n_components)
        store_pca = pca.fit_transform(df_store_bis)
        cols = [str(i) + 'Component Store PCA' for i in range(1, n_components+1)]
        store_pca = pd.DataFrame(store_pca, columns = cols, index = df_store.index)
        stores = pd.Series(range(1, df_store.shape[0] + 1), dtype = 'float32')
        store_pca.insert(0, column = 'Store', value = stores) 
        return store_pca


    def transform_one_df(self, X_df, is_store):
        self.get_colums(is_store)
        X_df = self.fill_na(X_df)
        if 'StateHoliday' in X_df.columns:
            X_df.StateHoliday = X_df.StateHoliday.astype(str)
        X_df = pd.get_dummies(X_df, columns=self.COLUMNS_DUMMY, drop_first=True)
        X_df = self.encode_cyclic_values(X_df)
        X_df = self.encode_temporal_values(X_df)
        X_df.drop(columns=self.COLUMNS_TO_DROP, axis=1, inplace=True)
        return X_df.astype('float32').fillna(0)

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

    def get_colums(self, is_store):
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
            self.COLUMNS_TO_DROP = []
        else:
            self.COLUMNS_DUMMY = [
                'StateHoliday'
            ]
            self.COLUMNS_CYCLIC = [
                'DayOfWeek'
            ]
            self.COLUMNS_TEMPORAL = {}
            self.COLUMNS_TO_DROP = [
                'Customers',
                'Date'
            ]


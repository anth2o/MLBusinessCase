from sklearn.model_selection import TimeSeriesSplit
from train import rmse, r2
import pandas as pd
import numpy as np
import preprocessing as pp
from model import *
from train import train_model
from predict import predict

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn



'''In order to evaluate our model, we propose two pipelines :
        - a classic evaluation with one train set and one test set where the test set is 6-weeks long
        - a kfold evaluation where the dates order is respected : we always evaluate our model on a test set which occured later
        than the train set. That allows our model not to be fitted on data in the future.
    '''


def cv_kfold(df, df_store, n_splits = 10, test_size = 42, with_pca = True):
    train = df.copy().iloc[::-1]
    train.Date = pd.to_datetime(train.Date)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    reg_model = Regressor()
    rmse_scores = []
    r2_scores = []

    date_grouping = train.groupby(train.Date)['Store']
    date_list = [g[0] for g in list(date_grouping)[:]]
    for train_index, test_index in tscv.split(date_grouping):
        # Fixed test set cardinality (in number of days)
        train_index = np.append(train_index, list(range(len(train_index), 1 + int(test_index[-1] - test_size))))
        test_index = test_index[(1 + int(train_index[-1] - test_index[0])):]
        
        train_dates = [date_list[train_index[0]], date_list[train_index[-1]]]
        test_dates = [date_list[test_index[0]], date_list[test_index[-1]]]
        train_mask = (train.Date >= train_dates[0]) & (train.Date <= train_dates[1])
        test_mask = (train.Date >= test_dates[0]) & (train.Date <= test_dates[1])
        
        # Train and test sets
        X_train, y_train, X_PCA_train = pp.Preprocessor().transform(df_store, train.loc[train_mask])
        X_test, y_test, X_PCA_test = pp.Preprocessor().transform(df_store, train.loc[test_mask])
        if with_pca :
            X_train = X_PCA_train.copy()
            X_test = X_PCA_test.copy()
        # Dummy variables can induce differences in the schemas
        missing_test = set(X_train.columns) - set(X_test.columns)
        missing_train = set(X_test.columns) - set(X_train.columns)
        for c in missing_test:
            X_test[c] = 0
        for c in missing_train:
            X_train[c] = 0
        # Reorder to match columns order in train and test
        X_test = X_test[X_train.columns]
        
        # Model fitting on training set
        train_model(reg_model, X_train, y_train)

        # Scoring on test set
        y_pred = reg_model.predict(X_test)
        rmse_scores.append(rmse(y_test, y_pred))
        r2_scores.append(r2(y_test, y_pred))
            
    # Final display
    for i in range(n_splits):
        print("FOLD " + str(i + 1) + ": " + "RSME = " + str(rmse_scores[i]) + 
        " | R² = " + str(r2_scores[i]))
    results = {}
    results['RMSE'] = rmse_scores
    results['R2'] = r2_scores
    # Overall scores
    w = [1 + 0.5 * i for i in range(1, n_splits + 1)]
    print("--- OVERALL ---")
    print("RSME = " + '{0:.2f}'.format(np.average(rmse_scores, weights=w)) + " | R² = " + '{0:.2f}'.format(np.average(r2_scores, weights=w)))
    return


def cv_1fold(df, df_store, with_pca = True):
    train = df.copy().iloc[::-1]
    train.Date = pd.to_datetime(train.Date)
    train_set = train[train.Date < '2015-06-19']
    test_set = train[train.Date >= '2015-06-19']
    reg_model = Regressor()
    X_train, y_train, X_PCA_train = pp.Preprocessor().transform(df_store, train_set)
    X_test, y_test, X_PCA_test = pp.Preprocessor().transform(df_store, test_set)
    # Dummy variables can induce differences in the schemas
    if with_pca :
        X_train = X_PCA_train.copy()
        X_test = X_PCA_test.copy()
    missing_test = set(X_train.columns) - set(X_test.columns)
    missing_train = set(X_test.columns) - set(X_train.columns)
    for c in missing_test:
        X_test[c] = 0
    for c in missing_train:
        X_train[c] = 0
    # Reorder to match columns order in train and test
    X_test = X_test[X_train.columns]
    # Model fitting on training set
    train_model(reg_model, X_train, y_train)
    # Scoring on test set
    y_pred = reg_model.predict(X_test)
    rmse_scores = rmse(y_test, y_pred)
    r2_scores = r2(y_test, y_pred)
    print("RSME = " + str(rmse_scores) + " | R² = " + str(r2_scores))
    results = {}
    results['RMSE'] = rmse_scores
    results['R2'] = r2_scores
    return results


'Evaluating the model : '
if __name__ == "__main__":
    df_store = pd.read_csv('../data/store.csv')
    df_train = pd.read_csv('../data/train.csv', low_memory=False)
    cv_kfold(df_train, df_store)
    
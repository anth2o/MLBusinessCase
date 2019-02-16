import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
    
# Model training and scoring
def train_model(model, X_df, y):
    """
    Function that trains a model and scores it
    using a cross-validation scheme
    """
    n_splits = 10
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    date_grouping = X_df.groupby(X_df.Date)['Store']
    date_list = [g[0] for g in list(date_grouping)[:]]
    for train_index, test_index in tscv.split(date_grouping):
        # Date lists corresponding to splits
        train_dates = [date_list[train_index[0]], date_list[train_index[-1]]]
        test_dates = [date_list[test_index[0]], date_list[test_index[-1]]]
        
        train_mask = (X_df.Date >= train_dates[0]) & (X_df.Date <= train_dates[1])
        test_mask = (X_df.Date >= test_dates[0]) & (X_df.Date <= test_dates[1])
        
        X_train = X_df.loc[train_mask].drop('Date', axis=1)
        y_train = y.loc[train_mask].drop('Date', axis=1)
        X_test = X_df.loc[test_mask].drop('Date', axis=1)
        y_test = y.loc[test_mask].drop('Date', axis=1)
        
        # Model fitting on training set
        model.fit(X_train, y_train)
        
        # Scoring on test set
        rmse_scores.append(rmse(y_test, model.predict(X_test)))
        r2_scores.append(r2(y_test, model.predict(X_test)))
        
    # Final display
    for i in range(n_splits):
        print("FOLD " + str(i + 1) + ": " + "RSME = " + str(rmse_scores[i]) + 
          " | R² = " + str(r2_scores[i]))
    

# Scoring metrics
def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def r2(y, y_pred):
    return r2_score(y, y_pred)
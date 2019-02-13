import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
    
# Model training and scoring
def train_model(model, X_df, y):
    """
    Function that trains a model and scores it
    using a cross-validation scheme
    """
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmse_scores = []
    r2_scores = []
    for train_index, test_index in tscv.split(X_df):
       X_train, X_test = X_df.iloc[train_index], X_df.iloc[test_index]
       y_train, y_test = y.iloc[train_index], y.iloc[test_index]
       model.fit(X_train, y_train)
       rmse_scores.append(rmse(y_test, model.predict(X_test)))
       r2_scores.append(r2(y_test, model.predict(X_test)))
    for i in range(n_splits):
        print("FOLD " + str(i) + ": " + "RSME = " + str(rmse_scores[i]) + 
              " | R² = " + str(r2_scores[i]))

# Scoring metrics
def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def r2(y, y_pred):
    return r2_score(y, y_pred)
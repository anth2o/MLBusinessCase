import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Model training and scoring


def train_model(model, X_df, y):
    model.fit(X_df.loc[:, X_df.columns != 'Date'], y)

# Scoring metrics


def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


def r2(y, y_pred):
    return r2_score(y, y_pred)

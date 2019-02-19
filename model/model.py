# Regression imports
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor

# =============================================================================
# REGRESSION MODEL CLASS
# =============================================================================

class Regressor(BaseEstimator):
    def __init__(self):
        # We used a Grid search to estimate good hyperparameters for the RandomForestRegressor
        rf_regressor = RandomForestRegressor()        
# =============================================================================
#         param_grid = {
#             'max_depth': [3, 5, 10],
#             'n_estimators': [10, 100]
#         }
#         grid_search = GridSearchCV(rf_regressor, param_grid, scoring='r2', cv=2, verbose=10)
# =============================================================================
        self.model = make_pipeline(StandardScaler(), rf_regressor)

    # Fit inner model to the training data
    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    # Target prediction for given inputs
    def predict(self, X):
        # Sales cannot be negative
        return np.maximum(self.model.predict(X.loc[:,X.columns != 'Date']), 0)

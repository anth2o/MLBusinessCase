def predict(model, X_df):
    """
    Function that predicts labels based on inputs and a given model
    """
    return model.predict(X_df.loc[:, X_df.columns != 'Date'])
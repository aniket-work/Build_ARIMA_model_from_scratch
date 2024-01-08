import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def fit_MA_model(q, residuals):
    """
    Fit Moving Average (MA) model using residuals.

    Args:
    - q (int): The order of the MA model.
    - residuals (pandas.DataFrame): DataFrame containing residuals.

    Returns:
    - list: Results including trained data, test data, coefficients, intercept, and RMSE.
    """
    # Creating shifted values based on the order of the MA model
    for i in range(1, q+1):
        residuals[f'Shifted_values_{i}'] = residuals['Residuals'].shift(i)

    train_size = int(0.8 * residuals.shape[0])

    # Splitting data into training and test sets
    train_residuals = residuals.iloc[:train_size, :]
    test_residuals = residuals.iloc[train_size:, :]

    # Dropping NaN values and preparing data for training
    train_residuals_clean = train_residuals.dropna()
    X_train = train_residuals_clean.iloc[:, 1:].values.reshape(-1, q)
    y_train = train_residuals_clean.iloc[:, 0].values.reshape(-1, 1)

    # Fitting a linear regression model
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Calculating model parameters
    theta = lr.coef_.T
    intercept = lr.intercept_
    train_residuals_clean['Predicted_Values'] = X_train.dot(lr.coef_.T) + lr.intercept_

    # Plotting the predicted values against residuals for the test set
    X_test = test_residuals.iloc[:, 1:].values.reshape(-1, q)
    test_residuals['Predicted_Values'] = X_test.dot(lr.coef_.T) + lr.intercept_
    test_residuals[['Residuals', 'Predicted_Values']].plot()

    # Calculating RMSE for the test set
    RMSE = np.sqrt(mean_squared_error(test_residuals['Residuals'], test_residuals['Predicted_Values']))

    # Displaying RMSE and the value of q
    print(f"The RMSE is: {RMSE}, Value of q: {q}")

    return [train_residuals_clean, test_residuals, theta, intercept, RMSE]

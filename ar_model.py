import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def generate_lagged_terms(p, dataframe):
    """
    Generate lagged terms in the DataFrame based on the specified order (p).

    Args:
    - p (int): The order of lagged terms.
    - dataframe (pandas.DataFrame): Input DataFrame containing the 'Value' column.

    Returns:
    - pandas.DataFrame: DataFrame with additional columns of lagged values.
    """
    df_temp = dataframe.copy()

    # Generating the lagged p terms
    for i in range(1, p+1):
        df_temp[f'Shifted_values_{i}'] = df_temp['Value'].shift(i)

    return df_temp

def split_train_test(dataframe, p):
    """
    Split the dataset into training and test sets based on lagged values.

    Args:
    - dataframe (pandas.DataFrame): DataFrame with lagged values.
    - p (int): The order of lagged terms.

    Returns:
    - tuple: DataFrames for training and test sets.
    """
    train_size = int(0.8 * dataframe.shape[0])

    df_train = dataframe.iloc[:train_size, :]
    df_test = dataframe.iloc[train_size-p:, :]  # Consider lagged values for test set

    df_train_2 = df_train.dropna()

    return df_train_2, df_test



def train_AR_model(df_train, p):
    """
    Train an AutoRegressive (AR) model using linear regression.

    Args:
    - df_train (pandas.DataFrame): DataFrame for training the model.
    - p (int): The order of lagged terms.

    Returns:
    - tuple: Model coefficients, intercept, and predicted values for the training set.
    """
    X_train = df_train.iloc[:, 1:].values.reshape(-1, p)
    y_train = df_train.iloc[:, 0].values.reshape(-1, 1)

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    theta = lr.coef_.T
    intercept = lr.intercept_

    df_train['Predicted_Values'] = X_train.dot(lr.coef_.T) + lr.intercept_

    return theta, intercept, df_train

def test_AR_model(df_test, lr_model, intercept, p):
    """
    Test the AutoRegressive (AR) model on the test set.

    Args:
    - df_test (pandas.DataFrame): DataFrame for testing the model.
    - lr_model: Trained linear regression model.
    - intercept: Intercept obtained from training.
    - p (int): The order of lagged terms.

    Returns:
    - float: Root Mean Squared Error (RMSE) for the test set.
    """
    X_test = df_test.iloc[:, 1:].values.reshape(-1, p)
    df_test['Predicted_Values'] = X_test.dot(lr_model) + intercept

    RMSE = np.sqrt(mean_squared_error(df_test['Value'], df_test['Predicted_Values']))

    return RMSE

def AR_model(p, dataframe):
    """
    Perform AutoRegressive (AR) modeling on the provided DataFrame.

    Args:
    - p (int): The order of lagged terms.
    - dataframe (pandas.DataFrame): Input DataFrame with 'Value' column.

    Returns:
    - list: Results including trained data, test data, coefficients, intercept, and RMSE.
    """
    df_with_lagged_values = generate_lagged_terms(p, dataframe)
    df_train, df_test = split_train_test(df_with_lagged_values, p)
    
    theta, intercept, trained_data = train_AR_model(df_train, p)
    RMSE = test_AR_model(df_test, theta, intercept, p)
    
    print(f"The RMSE is: {RMSE}, Value of p: {p}")
    
    return [trained_data, df_test, theta, intercept, RMSE]

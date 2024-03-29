import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from ar_model import AR_model
from ma_model import fit_MA_model

# Part 1 : understand data 

file_path = "data/Netflix_stock_history.csv"
output_folder = "results"

# Load the data
df = pd.read_csv(file_path, index_col="Date")

# Reset index to make 'Date' a regular column
df = df.reset_index()

# Create the 'results' directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Create a figure and an axis object for the plot
fig, ax = plt.subplots()

# Plot the close prices on the axis
ax.plot(df["Close"])

# Set the title and labels for the plot
ax.set_title("Netflix Stock Close Prices")
ax.set_xlabel("Date")
ax.set_ylabel("Close Price")

# Save the figure as a png file in the output folder
fig.savefig(f"{output_folder}/netflix_stock.png")

# Close the figure
plt.close(fig)

# Part 2 : decompose data 

def process_stock_data(df):
    # Copy the DataFrame to avoid modifying the original
    processed_df = df.copy()

    # Define columns to drop
    cols_to_drop = ['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits']

    # Drop specified columns
    processed_df.drop(cols_to_drop, axis=1, inplace=True)

    # Convert 'Close' column to numeric
    processed_df['Close'] = pd.to_numeric(processed_df['Close'], errors='coerce')

    # Drop rows with missing 'Close' values, if any
    processed_df = processed_df.dropna(subset=['Close'])

    return processed_df



filtered_dataframe = process_stock_data(df)
filtered_dataframe = filtered_dataframe.groupby('Date')['Close'].sum().reset_index()
filtered_dataframe.head()

# Data decomposition
from pylab import rcParams
rcParams['figure.figsize'] = 18, 8

# Convert 'Date' column to datetime if not already in datetime format
filtered_dataframe['Date'] = pd.to_datetime(filtered_dataframe['Date'])

# Set 'Date' column as the index
filtered_dataframe.set_index('Date', inplace=True)

# Now perform seasonal decomposition
# Now perform seasonal decomposition
decomposition = sm.tsa.seasonal_decompose(filtered_dataframe['Close'], model='additive', period=30)  # Assuming monthly data

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Plot the decomposed series
plt.figure(figsize=(18, 8))
plt.subplot(411)
plt.plot(filtered_dataframe.index, filtered_dataframe['Close'], label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(filtered_dataframe.index, trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(filtered_dataframe.index, seasonal, label='Seasonal')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(filtered_dataframe.index, residual, label='Residual')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig("results/decomposition.png")


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def run_adf(ts_data):
    result = adfuller(ts_data)
    labels = ['ADF Test Statistic', 'p-value', 'Number of Lags Used', 'Number of Observations Used']

    plt.figure(figsize=(8, 6))

    # Extract ADF Test Statistic and p-value
    adf_statistic = result[0]
    p_value = result[1]

    # Plotting ADF Test Statistic and p-value
    plt.bar(['ADF Test Statistic', 'p-value'], [adf_statistic, p_value], color=['blue', 'green'])
    plt.title('Results of the Augmented Dickey-Fuller Test')
    plt.ylabel('Value')
    plt.xlabel('Test')

    # Save the figure as a png file
    plt.savefig("results/adf_statistics.png")
    
    return adf_statistic, p_value

ts_data = np.log(filtered_dataframe['Close']).diff().dropna().values.flatten()
adf_statistic, p_value = run_adf(ts_data)

# Convert ts_data back to a pandas Series
ts_series = pd.Series(ts_data)

# Plot ACF
plt.figure(figsize=(10, 6))
plot_acf(ts_series, lags=50)
plt.title('Autocorrelation Function (ACF)')
plt.xlabel('Lag')
plt.ylabel('ACF')
plt.savefig("results/acf_plot.png")

# Plot PACF
plt.figure(figsize=(10, 6))
plot_pacf(ts_series, lags=50)
plt.title('Partial Autocorrelation Function (PACF)')
plt.xlabel('Lag')
plt.ylabel('PACF')
plt.savefig("results/pacf_plot.png")

# Part 3 Fit AR(p)
ts_dataframe = pd.DataFrame({'Value': ts_data})  # Creating a DataFrame with 'Value' column

best_RMSE = float('inf')  # Setting initial best_RMSE to infinity or any high value
best_p = -1

for i in range(1, 26):  # Loop from 1 to 25
    [df_train, df_test, theta, intercept, RMSE] = AR_model(i, pd.DataFrame(ts_dataframe))

    if RMSE < best_RMSE:
        best_RMSE = RMSE
        best_p = i

print(f"The best value of p is {best_p} with RMSE: {best_RMSE}")

df_c = pd.concat([df_train, df_test])
df_c[['Value', 'Predicted_Values']].plot()

# Save the plot as ar_model_run_op.png
plt.title('AR Model: Actual vs. Predicted Values')
plt.xlabel('Time')
plt.ylabel('Value')
plt.savefig("results/ar_model_run_op.png")

# Part 4 Fit MA

# Calculate residuals for the MA model
residuals = df_c['Value'] - df_c['Predicted_Values']
residuals_df = pd.DataFrame({'Residuals': residuals})

# Plot the density (KDE) of the residuals
residuals_df.plot(kind='kde', title='Density Plot of Residuals for MA Model')
plt.xlabel('Residuals')
plt.ylabel('Density')
plt.savefig("results/ma_residuals_density_plot.png")

best_RMSE = float('inf')
best_q = -1

for i in range(1, 13):
    [res_train, res_test, theta, intercept, RMSE] = fit_MA_model(i, pd.DataFrame({'Residuals': residuals}))

    
    if RMSE < best_RMSE:
        best_RMSE = RMSE
        best_q = i

print(f"The best value of q is {best_q} with RMSE: {best_RMSE}")

res_c = pd.concat([res_train, res_test])
res_c[['Residuals', 'Predicted_Values']].plot()

# Save the plot as ar_model_run_op.png
plt.title('MA Model: Actual vs. Predicted Values')
plt.xlabel('Time')
plt.ylabel('Value')
plt.savefig("results/ma_model_run_op.png")











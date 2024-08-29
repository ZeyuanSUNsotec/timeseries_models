import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from timesfm import TimesFm
### use timesfm to predict stock price


## stock code 
#codelist = ["000001.ss"]
#codelist = ["RHM.DE"]
#codelist = ["AAPL"]
#codelist = ["TSLA"]
tickerSymbol = 'MBG.DE'

# Get data on this ticker
tickerData = yf.Ticker(tickerSymbol)
#earnings = tickerData.earnings
#print("earnings", earnings)

# Define the start and end dates for potential context
start_date = datetime.datetime(2024, 5, 1, 0, 0, 0)
end_date = datetime.datetime(2024, 8, 1, 0, 0, 0)

# Get historical prices 
tickerDf = tickerData.history(start=start_date, end=end_date, interval='1h')
# get hourly high
data2 = tickerDf['High'].dropna()

## set parameters for the timesfm model
context_len = 512  # set max. context, need to be multiple of 32, currently max. 512 
horizon_len = 128  # largest horizon length needed in forecasting, recommend horizon length <= context length

## input/training data is the context 
forecast_input = [data2.values]
# get current time
current_datetime = datetime.datetime.now()  # Get current date and time
# test data
tickerDf_curr = tickerData.history(start=end_date, end=current_datetime, interval='1h')
print("history", tickerDf_curr.shape)
data_recent = tickerDf_curr['High'].dropna()

next_hour = (end_date + pd.Timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
forecast_dates = pd.date_range(start=next_hour, periods=horizon_len, freq='h')

## initialize the model and load the checkpoint
tfm = TimesFm(
    context_len=context_len,
    horizon_len=horizon_len,
    input_patch_len=32,
    output_patch_len=128,
    num_layers=20,
    model_dims=1280,
    backend='cpu',  
)
tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")
## prediction
# mode 0, high frequency
point_forecast, experimental_quantile_forecast = tfm.forecast(
    forecast_input,
    freq=[0],
)
# transform output into a timeseries
forecast_series = pd.Series(point_forecast[0], index=forecast_dates)

plt.subplot(3,1,1)
plt.xlabel("Date")
plt.ylabel("Price")
data_recent.plot(label="Actual Prices (2024-Now)")
forecast_series.plot(label="Forecasted Prices (2024-Now)")
plt.title("mode 0")
plt.legend()

# mode 1, medium frequency
point_forecast, experimental_quantile_forecast = tfm.forecast(
    forecast_input,
    freq=[0],
)
forecast_series = pd.Series(point_forecast[0], index=forecast_dates)
plt.subplot(3,1,2)
plt.xlabel("Date")
plt.ylabel("Price")
data_recent.plot(label="Actual Prices (2024-Now)")
forecast_series.plot(label="Forecasted Prices (2024-Now)")
plt.title("mode 1")
plt.legend()

# mode 2, low frequency
point_forecast, experimental_quantile_forecast = tfm.forecast(
    forecast_input,
    freq=[2],
)
forecast_series = pd.Series(point_forecast[0], index=forecast_dates)
plt.subplot(3,1,3)
plt.xlabel("Date")
plt.ylabel("Price")
data_recent.plot(label="Actual Prices (2024-Now)")
forecast_series.plot(label="Forecasted Prices (2024-Now)")
plt.title("mode 2")
plt.legend()

plt.show()  
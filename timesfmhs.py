import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from datetime import date
from timesfm import TimesFm
from huggingface_hub import login

# set begin and end for training set
start = date(2020, 1, 1)  # use date class to create date
end = date(2024, 1, 1) 
#get training data
codelist = ["000001.ss"]
#codelist = ["RHM.DE"]
#codelist = ["AAPL"]

# in case downloading data fails, try max. three times
for retry in range(3):
    try:
        data2 = yf.download(codelist, start=start, end=end)
        data2 = data2['Adj Close'].dropna()  # delete empty values
        if not data2.empty:
            break  # jump out if succed
    except Exception as e:
        print(f"download fails，the {retry+1}try. Error：{e}")
        if retry < 2:  # wait before last try
            time.sleep(5)  # wait for 5 sec to retry

if data2.empty:
    raise ValueError("Empty data, check internet connection")

context_len = 512  # set max. context, need to be multiple of 32, currently max. 512 
horizon_len = 128  # largest horizon length need in forecasting, recommend horizon length <= context length

if len(data2) < context_len:
    raise ValueError(f"data length shorter than context length（{context_len}）")

context_data = data2[-context_len:]  # use the latest [context_len] days as context 

# initialize and load the model
tfm = TimesFm(
    context_len=context_len,
    horizon_len=horizon_len,
    input_patch_len=32,
    output_patch_len=128,
    num_layers=20,
    model_dims=1280,
    backend='cpu',  
)

# load the currently provided checkpoint
tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")

# prepare the data
forecast_input = [context_data.values]

# do prediction
point_forecast, experimental_quantile_forecast = tfm.forecast(
    forecast_input,
    freq=[0],
)

forecast_dates = pd.date_range(start=data2.index[-1] + pd.Timedelta(days=1), periods=horizon_len, freq='B')
forecast_series = pd.Series(point_forecast[0], index=forecast_dates)

# predicted data
current_date = datetime.datetime.now().date()
data_recent = yf.download(codelist, start=date(2024, 1, 1), end=current_date)
#mode 0
plt.subplot(3,1,1)
plt.xlabel("Date")
plt.ylabel("Price")
#plt.plot(input_x_0.reshape(1,-1), input_y_0.reshape(1,-1),"b.", label="real values")
data_recent = data_recent['Adj Close'].dropna()
plt.plot(data_recent.index, data_recent.values, label="Actual Prices (2024-Now)")
#plt.plot(x_0.reshape(1,-1), point_forecast[0].reshape(1,-1),"g.", label="predicted values")
plt.plot(data2.index, data2.values, label="Actual Prices")
plt.plot(forecast_series.index, forecast_series.values, label="Forecasted Prices")
plt.title("mode 0")
plt.legend()

#mode 1
point_forecast, experimental_quantile_forecast = tfm.forecast(
    forecast_input,
    freq=[1],
)
forecast_series = pd.Series(point_forecast[0], index=forecast_dates)
plt.subplot(3,1,2)
plt.xlabel("Date")
plt.ylabel("Price")
plt.plot(data_recent.index, data_recent.values, label="Actual Prices (2024-Now)")
#plt.plot(x_0.reshape(1,-1), point_forecast[0].reshape(1,-1),"g.", label="predicted values")
plt.plot(data2.index, data2.values, label="Actual Prices")
plt.plot(forecast_series.index, forecast_series.values, label="Forecasted Prices")
plt.title("mode 1")
plt.legend()

#mode 2
point_forecast, experimental_quantile_forecast = tfm.forecast(
    forecast_input,
    freq=[2],
)
forecast_series = pd.Series(point_forecast[0], index=forecast_dates)
plt.subplot(3,1,3)
plt.xlabel("Date")
plt.ylabel("Price")
plt.plot(data_recent.index, data_recent.values, label="Actual Prices (2024-Now)")
#plt.plot(x_0.reshape(1,-1), point_forecast[0].reshape(1,-1),"g.", label="predicted values")
plt.plot(data2.index, data2.values, label="Actual Prices")
plt.plot(forecast_series.index, forecast_series.values, label="Forecasted Prices")
plt.title("mode 2")
plt.legend()

plt.show()  

# set plot size
"""
# save plot
plt.savefig(f'{codelist[0]}_comparison.png', bbox_inches='tight') 

# close plot
#plt.close(fig='all')
plt.show()"""
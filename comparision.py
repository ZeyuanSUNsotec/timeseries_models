import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import timesfm

from darts import TimeSeries
# mae: if data contains outliers. 
# rmse: sensitive to large errors. 
# mse: sensitive to outliers. 
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_squared_error

### This script compares the behavior of TimesFM, seasonal ARIMA VAR and ARIMAX on FlightPassengerData and Housing


class Models:

    def __init__(self, df, full_len=600, c_len=480, max_lag=36, goal=None):
        # only acctept dataframe
        self.df = df
        self.full_len = full_len # length of dataset to be used
        # context length, multiplier of 32
        self.c_len = c_len  
        # horizon length
        self.horizon = full_len-c_len  
        # number of lagged observations that the model determined to use based on the fitting process
        self.max_lag = max_lag  
        self.goal = goal

    def data_4_tf(self):
        # transform data into array format for TimesFM
        if not self.goal is None:
            arr = (self.df[self.goal].values).reshape(-1,)  # from dataframe into numpy array
            train, val = arr[:self.c_len], arr[self.c_len:]
        else:
            arr = (self.df.values).reshape(-1,)  # from pandas dataframe into numpy array
            train, val = arr[:self.c_len], arr[self.c_len:]
        return train, val
    
    def data_4_VAR(self):
        train, test = self.df[:self.c_len], self.df[self.c_len:]
        return train, test
    
    def data_4_ARIMAX(self): 
        # load data and build dataframe for ARIMAX(standard ARIMA considers Exogenous data)
        train, test = self.df[:self.c_len], self.df[self.c_len:]
        endog_train = train[[self.goal]]
        exog_train = train.drop(columns=[self.goal])

        endog_test = test[[self.goal]]
        exog_test = test.drop(columns=[self.goal])
        return endog_train, exog_train, endog_test, exog_test

    def data_4_ARIMA_SEASONAL(self, series):
        # input for ARIMA Seasonal must be a single TimeSeries
        train, test = series[:self.c_len], series[self.c_len:]
        return train, test

    def predict_timesfm(self, train, frequency=2):
        # train: training data
        tfm = timesfm.TimesFm(
            #context_len:multiplier of input_patch_len
            context_len=self.c_len,
            #horizon_len:largest horizon length needed in the forecasting task
            horizon_len=self.horizon, 
            input_patch_len=32,  #fix
            output_patch_len=128,  #fix
            num_layers=20,  #fix
            model_dims=1280,  #fix
            #backend: cpu/gpu/tpu
            backend="cpu",
        )
        tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")
        forecast_input = [
            train,
        ]
        frequency_input = [frequency]
        # if use Dataframe instead of Array, tfm.forecast_on_df is needed, "unique_id" col is obligated in df
        point_forecast, experimental_quantile_forecast = tfm.forecast(
            forecast_input,
            freq=frequency_input,
        )
        return point_forecast
    
    def predict_ARIMA_SEASONAL(self, train):
        from darts.models import AutoARIMA
        ## Univariate comparision->seasonal ARIMA
        # m=12 for monthly data to capture yearly seasonality
        # relate to file "parameter_search" in case seasonal is set to false
        model = AutoARIMA(seasonal=True, m=12)  # m is the length of period
        model.fit(train)
        forecast = model.predict(self.horizon)
        return forecast

    def predict_ARIMAX(self, endog_train, exog_train, exog_test, p, q, d):
        import statsmodels.api as sm
        """
        p (autoregressive order): The number of lag observations included in the model. Here, p=1 means the model includes one lag of the dependent variable.
        d (differencing order): The number of times the data has had past values subtracted. Here, d=0 indicates no differencing, implying the data is already stationary.
        q (moving average order): The size of the moving average window. Here, q=1 means the model includes one lag of the error term.
        """
        arimax_model = sm.tsa.statespace.SARIMAX(endog=endog_train, exog=exog_train, 
                                                 order=(int(p), int(q), int(d))).fit(disp=0)
        pred_result = arimax_model.forecast(self.horizon, exog=exog_test)
        return pred_result

    def predcit_VAR(self, train, test):
        """
        Inputs:
            dataframe of training and test data
        return:
            dataframe of predicion
        """
        from statsmodels.tsa.api import VAR
        model = VAR(train)
        # Fit the model-> find out the best lag within max_lag with the method "aic"
        fitted_model = model.fit(maxlags=self.max_lag, ic='aic')  
        # Predict on the test set
        lag_order = fitted_model.k_ar  #  retrieves the number of lags (time steps) used by the fitted VAR model.
        forecast_input = train.values[-lag_order:]  # context length
        forecast = fitted_model.forecast(y=forecast_input, steps=len(test))
        # Convert forecast to DataFrame
        forecast_df = pd.DataFrame(forecast, index=test.index, columns=test.columns)
        return forecast_df, lag_order

# True for comparision of on univariate dataset
# False for comparision of on multivariate dataset
univariate = False
case = "aps"
if univariate:
    # import dataset
    from darts.datasets import AirPassengersDataset  #aps, monthly
    from read_tsf import t1_timeseries as australian_ele  # ae half_hourly

    if case == "aps":
        # in case of airpassenger
        series = AirPassengersDataset().load()
        df = series.pd_dataframe()  # from timeseries into pandas dataframe
    #   series.plot() # show this air passenger plot, we can see it is seasonal, 144 oberveations in amount
        c_len = 96  # context length, multiplier of 32 (for timesfm)
        frequency = 1
        horizon = len(series)-c_len  # prediction horizon
        # define model complex
        models =  Models(df, len(series), c_len)
        series_arima = series

    elif case == "ae":
        # in case of australian electricity price
        full_len = 500  #150
        c_len = 320  # context length, multiplier of 32 (for timesfm)
        frequency = 0
        series = australian_ele[0:full_len]
        df = series["value"]
        horizon = full_len-c_len
        # define model complex
        models =  Models(df, full_len, c_len)
        # data formalized for seasonal arima
        series_arima = TimeSeries.from_dataframe(series.reset_index(), time_col='timestamp', value_cols='value')

    ## Univariate Comparision->TimesFM
    train, val = models.data_4_tf()
    point_forecast = models.predict_timesfm(train, frequency=frequency)
    mae = mean_absolute_error(val.reshape(-1,), point_forecast.reshape(-1,))
    print("timesfm mae:", mae)

    # Plot
    x_1 = np.linspace(0, len(series), len(series)).reshape(-1,)  # training data x-axis
    x_2 = np.linspace(c_len, len(series), len(series)-c_len).reshape(-1,)  # prediction x-axis
    plt.plot(x_1, (df.values).reshape(-1,), label="real values")
    plt.plot(x_2, point_forecast.reshape(-1,), label="timesfm")

    ## Univariate comparision->seasonal ARIMA
    train, test = models.data_4_ARIMA_SEASONAL(series_arima)
    forecast = models.predict_ARIMA_SEASONAL(train)
    error = mean_absolute_error(test.values(), forecast.values())
    print(f'seasonal ARIMA mae: {error:.2f}')

    # Plot
    plt.plot(x_2, forecast.values().reshape(-1,), label="seasonal ARIMA")
    
    plt.title("Comparision of timesfm and seasonal ARIMA")
    plt.legend()
    plt.text(0, 8, f'TimesFM mae: {mae:.2f}')
    plt.text(20, 8, f'seasonal ARIMA mae: {error:.2f}')
else:
    from sklearn.datasets import fetch_california_housing  # multivariate data
    # set hyperparameters
    full_len = 600 # length of dataset to be used
    c_len = 480  # context length, multiplier of 32 (for Timesfm)
    horizon = full_len-c_len  # horizon length
    max_lag = 36  # number of lagged observations that the model determined to use based on the fitting process
    
    # load data
    data = fetch_california_housing()
    # read data into Dataframe
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['MedHouseVal'] = data.target  # target value
    df = df[:full_len]  # use only part of the whole set
    
    # define model complex
    models = Models(df, full_len, c_len, max_lag, goal='MedHouseVal')

    ## Multivariate Comparision->TimesFM
    train, val = models.data_4_tf()
    point_forecast = models.predict_timesfm(train, frequency=2)
    mae = mean_absolute_error(val.reshape(-1,), point_forecast.reshape(-1,))
    rmse_timesfm = root_mean_squared_error(val.reshape(-1,), point_forecast.reshape(-1,))
    mse_timesfm = mean_squared_error(val.reshape(-1,), point_forecast.reshape(-1,))

    # Plot
    #x_1 = np.linspace(0, len(arr), len(arr)).reshape(-1,)
    x_2 = np.linspace(c_len, c_len+horizon, horizon).reshape(-1,)
    plt.plot(x_2, (df['MedHouseVal'].values).reshape(-1,)[-horizon:], label="real values")
    plt.plot(x_2, point_forecast.reshape(-1,), label="timesfm prediction")

    ## Multivariate Comparision->VAR
    train, test = models.data_4_VAR()
    forecast_df, lag_order = models.predcit_VAR(train, test)
    error = mean_absolute_error(test['MedHouseVal'], forecast_df['MedHouseVal'])
    rmse_var = root_mean_squared_error(test['MedHouseVal'], forecast_df['MedHouseVal'])
    mse_var = mean_squared_error(test['MedHouseVal'], forecast_df['MedHouseVal'])

    # Plot
    forecast_df['MedHouseVal'].plot(title="Comparision of different models on Multivariate data", 
                                    color="yellow", label="VAR prediction")
    train['MedHouseVal'][-lag_order:].plot(color="pink", label="VAR training data")

    ## Multivariate Comparision->ARIMA+endog data
    endog_train, exog_train, endog_test, exog_test = models.data_4_ARIMAX()
    pred_result = models.predict_ARIMAX(endog_train, exog_train, exog_test,
                                                  p=4, q=0, d=2)
    mae_arima = mean_absolute_error(endog_test, pred_result)
    rmse_arima = root_mean_squared_error(endog_test, pred_result)
    mse_arima = mean_squared_error(endog_test, pred_result)

    # Plot
    plt.plot(x_2, np.array(pred_result).reshape(-1,), label="ARIMAX prediction")
    
    plt.title("Comparision of different models on Multivariate data")
    plt.legend()
    #plt.text(480, 1, f'TimesFM mae: {mae:.2f}')
    #plt.text(480, 0.9, f'VAR mae: {error:.2f}')
    #plt.text(480, 0.8, f'ARIMAX mae: {mae_arima:.2f}')

    plt.text(480, 1, f'TimesFM mae: {mse_timesfm:.2f}')
    plt.text(480, 0.9, f'VAR mae: {mse_var:.2f}')
    plt.text(480, 0.8, f'ARIMAX mae: {mse_arima:.2f}')
    
plt.show()


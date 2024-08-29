import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import timesfm

from sklearn.metrics import mean_absolute_error

### This script compares the behavior of TimesFM, seasonal ARIMA VAR and ARIMAX on FlightPassengerData and Housing


class Models:

    def __init__(self, df, full_len=600, c_len=480, horizon = 120, max_lag=36, goal=None):
        # only acctept dataframe
        self.df = df
        self.full_len = full_len # length of dataset to be used
        # context length, multiplier of 32
        self.c_len = c_len  
        # horizon length
        self.horizon = horizon
        # number of lagged observations that the model determined to use based on the fitting process
        self.max_lag = max_lag  
        self.goal = goal

    def data_4_tf(self):
        # transform data into array format for TimesFM
        if not self.goal is None:
            arr = (self.df[self.goal].values).reshape(-1,)  # from dataframe into numpy array
            train, val = arr[:-self.horizon], arr[-self.horizon:]
        else:
            arr = (self.df.values).reshape(-1,)  # from pandas dataframe into numpy array
            train, val = arr[:self.c_len], arr[self.c_len:]
        return train, val
    
    def data_4_VAR(self):
        train, test = self.df[:self.c_len], self.df[self.c_len:]
        return train, test
    
    def data_4_ARIMAX(self): 
        # load data and build dataframe for ARIMAX(standard ARIMA considers Exogenous data)
        train, test = self.df[:-self.horizon], self.df[-self.horizon:]
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
        model = AutoARIMA(seasonal=True, m=12)  # m=12 for monthly data to capture yearly seasonality
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

if univariate:
    from darts.datasets import AirPassengersDataset
    series = AirPassengersDataset().load()
    #series.plot() # show this air passenger plot, we can see it is seasonal, 144 oberveations in amount
    c_len = 96  # context length, multiplier of 32 (for timesfm)
    horizon = len(series)-c_len  # prediction horizon

    df = series.pd_dataframe()  # from timeseries into pandas dataframe

    # define model complex
    models =  Models(df, len(series), c_len)

    ## Univariate Comparision->TimesFM
    train, val = models.data_4_tf()
    point_forecast = models.predict_timesfm(train, frequency=2)
    mae = mean_absolute_error(val.reshape(-1,), point_forecast.reshape(-1,))
    print("timesfm mae:", mae)

    # Plot
    x_1 = np.linspace(0, len(series), len(series)).reshape(-1,)
    x_2 = np.linspace(c_len, len(series), len(series)-c_len).reshape(-1,)
    plt.plot(x_1, (df.values).reshape(-1,), label="real values")
    plt.plot(x_2, point_forecast.reshape(-1,), label="timesfm")

    ## Univariate comparision->seasonal ARIMA
    train, test = models.data_4_ARIMA_SEASONAL(series)
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
    c_len = 512  # set max. context, need to be multiple of 32, currently max. 512 
    horizon = 256  # horizon length
    max_lag = 36  # number of lagged observations that the model determined to use based on the fitting process
    
    # load data
    #data = fetch_california_housing()
    # read data into Dataframe
    #df = pd.DataFrame(data.data, columns=data.feature_names)
    #df['MedHouseVal'] = data.target  # target value
    #df = df[:full_len]  # use only part of the whole set

    import yfinance as yf
    import datetime
    tickerSymbol = 'TL0.DE'
    # Get data on this ticker
    tickerData = yf.Ticker(tickerSymbol)

    # Define the start and end dates
    start_date = datetime.datetime(2024, 1, 1, 0, 0, 0)
    end_date = datetime.datetime(2024, 7, 1, 0, 0, 0)
    current_datetime = datetime.datetime.now()
    next_hour = (end_date + pd.Timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    forecast_dates = pd.date_range(start=next_hour, periods=horizon, freq='h')

    # Get all historical prices 
    tickerDf = tickerData.history(start=start_date, end=current_datetime, interval='1h')
    df = tickerDf#[-full_len:]
    val_data = df[-horizon:]
    val_data_df = pd.Series(val_data["High"].values, index=forecast_dates)
    val_data_df.plot(label="real data")
    # get hourly high
    #data2 = tickerDf['High'].dropna()
    
    # define model complex
    models = Models(df, full_len, c_len, horizon, max_lag, goal='High')#MedHouseVal

    ## Multivariate Comparision->TimesFM
    train, val = models.data_4_tf()
    point_forecast = models.predict_timesfm(train, frequency=2)
    #print("predicted length", point_forecast.shape)
    #print("validation length", val.shape)
    forecast_series = pd.Series(point_forecast[0], index=forecast_dates)
    forecast_series.plot(label="timesfm prediction")
    mae = mean_absolute_error(val.reshape(-1,), point_forecast.reshape(-1,))
    print("timesfm mae:", mae)

    # Plot
    #x_1 = np.linspace(0, len(arr), len(arr)).reshape(-1,)
    x_2 = np.linspace(c_len, c_len+horizon, horizon).reshape(-1,)
    #plt.plot(x_2, (df['High'].values).reshape(-1,)[-horizon:], label="real values")
    #plt.plot(x_2, point_forecast.reshape(-1,), label="timesfm prediction")
    """
    ## Multivariate Comparision->VAR
    train, test = models.data_4_VAR()
    forecast_df, lag_order = models.predcit_VAR(train, test)
    error = mean_absolute_error(test['High'], forecast_df['MedHouseVal'])
    print("VAR mae: ", error)

    # Plot
    forecast_df['High'].plot(title="Comparision of different models on Multivariate data", 
                                    color="yellow", label="VAR prediction")
    train['High'][-lag_order:].plot(color="pink", label="VAR training data")
    """
    ## Multivariate Comparision->ARIMA+endog data
    endog_train, exog_train, endog_test, exog_test = models.data_4_ARIMAX()
    #assert endog_train.shape==exog_train.shape, print(exog_train.shape, " and ", exog_train.shape)
    #assert endog_test.shape==exog_test.shape, print(exog_test.shape, " and ", exog_test.shape)
    pred_result = models.predict_ARIMAX(endog_train, exog_train, exog_test, p=4, q=0, d=2)
    f_d = pd.date_range(start=next_hour, periods=horizon, freq='h')
    arima_df = pd.Series(np.array(pred_result).reshape(-1,), index=f_d)
    arima_df.plot(title="Comparison",label="ARIMAX prediction")

    mae_arima = mean_absolute_error(endog_test, pred_result)
    # Plot
    #plt.plot(x_2, np.array(pred_result).reshape(-1,), label="ARIMAX prediction")
    print("ARIMA mae: ", mae_arima)
    
    #plt.title("Comparision of different models on Multivariate data")
    plt.legend()
    #plt.text(480, 1, f'TimesFM mae: {mae:.2f}')
    #plt.text(480, 0.9, f'VAR mae: {error:.2f}')
    #plt.text(480, 0.8, f'ARIMAX mae: {mae_arima:.2f}')
    
plt.show()
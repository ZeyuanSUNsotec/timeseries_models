import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.datasets import fetch_california_housing
import statsmodels.api as sm


# tune parameter for arima using grid search
full_len = 600 
c_len = 480  # multiplier of 32
horizon = full_len-c_len #################### data length wrong, fisrt cut it of,see line 41

# load data and build dataframe
def load_2_df(full_len, c_len):
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['MedHouseVal'] = data.target
    df = df[:full_len]   
    train, test = df[:c_len], df[c_len:]

    endog_train = train[['MedHouseVal']]
    exog_train = train.drop(columns=['MedHouseVal'])

    endog_test = test[['MedHouseVal']]
    exog_test = test.drop(columns=['MedHouseVal'])
    return endog_train, exog_train, endog_test, exog_test

def predict(p, q, d,endog_train, exog_train, exog_test, endog_test):
    # Initialize the model
    arimax_model = sm.tsa.statespace.SARIMAX(endog=endog_train, exog=exog_train, order=(int(p), int(q), int(d))).fit(disp=0)
    """
    p (autoregressive order): The number of lag observations included in the model. Here, p=1 means the model includes one lag of the dependent variable.
    d (differencing order): The number of times the data has had past values subtracted. Here, d=0 indicates no differencing, implying the data is already stationary.
    q (moving average order): The size of the moving average window. Here, q=1 means the model includes one lag of the error term.
    """
    pred_result = arimax_model.forecast(horizon, exog=exog_test)
    mae = mean_absolute_error(endog_test, pred_result)
    return mae, pred_result

def opt(p, q, d, full_len=full_len, c_len=c_len):
    endog_train, exog_train, endog_test, exog_test = load_2_df(full_len, c_len)
    mae, _ = predict(p, q, d,endog_train, exog_train, exog_test, endog_test)
    return mae

mae_init = 10000
a = {'p': 0, 'q': 0, 'd': 0}
for p in range(6):
    for q in range(6):
        for d in range(3):
            mae = opt(p, q, d, full_len=full_len, c_len=c_len)
            if mae < mae_init:
                mae_init = mae
                a['d'], a['p'], a['q'] = d, p, q

print("parameters", a)
print(mae_init)

endog_train, exog_train, endog_test, exog_test = load_2_df(full_len, c_len)
_, pred_result = predict(a['p'], a['q'], a['d'], endog_train, exog_train, exog_test, endog_test)
endog_test.plot(title="test data", color="black")
pred_result.plot(title="forecast data", color="green")
plt.legend()
plt.show()

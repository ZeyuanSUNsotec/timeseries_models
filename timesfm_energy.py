import pandas as pd
#from darts import TimeSeries
import timesfm
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("/mnt/c/Users/z.sun_sotec/timesfm/test/AirPassengers.csv", delimiter=",")
df = pd.read_csv("/mnt/c/Users/z.sun_sotec/timesfm/test/monthly-milk-production-pounds.csv", delimiter=",")
print(df.head(n=1))
# Create a TimeSeries, specifying the time and value columns
#series = TimeSeries.from_dataframe(df, "Month", "#Passengers")
# Set aside the last 36 months as a validation series
#train, val = series[:-64], series[-64:]

#passenger = df["#Passengers"]
passenger = df["Monthly milk production (pounds per cow)"]
train = passenger[:96].values

#val = passenger[96:].values
val = passenger[-80:].values
tfm = timesfm.TimesFm(
    #context_len:multiplier of input_patch_len
    context_len=96,
    #horizon_len:largest horizon length needed in the forecasting task
    horizon_len=80,#len(val),
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
    #np.sin(np.linspace(0, 20, 200)),
    #np.sin(np.linspace(0, 20, 400)),
]
frequency_input = [1] #[0, 1, 2]

point_forecast, experimental_quantile_forecast = tfm.forecast(
    forecast_input,
    freq=frequency_input,
)

x_1 = np.linspace(0, 96, 96)
y_1 = train
x_2 = np.linspace(96, 96+len(val), len(val))  #check how the model is defined
plt.plot(x_1.reshape(1,-1), y_1.reshape(1,-1),"b.", label="real values")
plt.plot(x_2.reshape(1,-1), point_forecast,"g.", label="predicted values")
plt.plot(x_2.reshape(1,-1), val.reshape(1,-1),"b.", label="true values")
plt.show()
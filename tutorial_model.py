import timesfm
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')


context_len = 512
horizon_len = 256
# initialize the model and load a checkpoint
tfm = timesfm.TimesFm(
    # set max. context, need to be multiple of 32, currently max. 512 
    context_len=context_len,
    # largest horizon length need in forecasting, recommend horizon length <= context length
    horizon_len=horizon_len,
    input_patch_len=32,  #fix
    output_patch_len=128,  #fix
    num_layers=20,  #fix
    model_dims=1280,  #fix
    #backend: cpu/gpu/tpu
    backend="cpu",
)
tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")

# training data
# examples with array inputs
input_x_0 = np.linspace(0, 20, 100)
input_x_1 = np.linspace(0, 20, 200)
input_x_2 = np.linspace(0, 20, 400)

input_y_0 = np.sin(input_x_0) + 1*np.random.rand(100)
input_y_1 = np.sin(input_x_1) + 1*np.random.rand(200)
input_y_2 = np.sin(input_x_2) + 1*np.random.rand(400)

# input_y = input_y.reshape(-1,1)
forecast_input = [
    input_y_0,
    input_y_1,
    input_y_2,
]

# could select three inputs which correspond to three modes for prediction horizon [0, 1, 2]
frequency_input = [0, 1, 2]
# 0 (default): high frequency, long horizon time series. 
# We recommend using this for time series up to daily granularity.

# 1: medium frequency time series. 
# We recommend using this for weekly and monthly data.

# 2: low frequency, short horizon time series. 
# We recommend using this for anything beyond monthly, e.g. quarterly or yearly.

#x,y
#x: mean forecast of size [inputs, forecast horizon]
#y: full forecast of size [inputs, forecast horizon, 1+quantiles]
point_forecast, experimental_quantile_forecast = tfm.forecast(
    forecast_input,
    freq=frequency_input,
)

x_0 = np.linspace(20, 20+0.2*horizon_len, horizon_len) #check how the model is defined
x_1 = np.linspace(20, 20+0.1*horizon_len, horizon_len)
x_2 = np.linspace(20, 20+0.05*horizon_len, horizon_len)

#mode 0
plt.subplot(3,1,1)
plt.plot(input_x_0.reshape(1,-1), input_y_0.reshape(1,-1),"b.", label="real values")
plt.plot(x_0.reshape(1,-1), point_forecast[0].reshape(1,-1),"g.", label="predicted values")
plt.title("mode 0")

#mode 1
plt.subplot(3,1,2)
plt.plot(input_x_1.reshape(1,-1), input_y_1.reshape(1,-1),"b.", label="real values")
plt.plot(x_1.reshape(1,-1), point_forecast[1].reshape(1,-1),"g.", label="predicted values")
plt.title("mode 1")

#mode 2
plt.subplot(3,1,3)
plt.plot(input_x_2.reshape(1,-1), input_y_2.reshape(1,-1),"b.", label="real values")
plt.plot(x_2.reshape(1,-1), point_forecast[2].reshape(1,-1),"g.", label="predicted values")
plt.title("mode 2")

#plt.legend()
plt.show()  
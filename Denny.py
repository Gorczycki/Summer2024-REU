import numpy as np
import matplotlib.pyplot as plt
import iisignature
import esig
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import pandas as pd
from bokeh.layouts import row
from bokeh.plotting import figure, show

# Function to simulate Brownian motion
def BM(N, t_final, t_0):
    time_grid = np.linspace(t_0, t_final, num=N)
    dt = time_grid[1] - time_grid[0]
    dB = np.random.normal(0, np.sqrt(dt), N)
    dB[0] = 0
    B = np.cumsum(dB)
    return time_grid, B

# Function to simulate stock prices using geometric Brownian motion
def Simulate_Stock(S_0, mu, sigma, N, t_final, t_0):
    time, B = BM(N, t_final, t_0)
    S = S_0 * np.exp((mu - 0.5 * sigma**2) * time + sigma * B)
    return S, B, time

def Signature(B_hat, N, order_model):
    nbr_components = B_hat.shape[1]
    keys=esig.sigkeys(nbr_components,order_model).strip().split(" ")
    S = np.array([100])
    sig_df = pd.DataFrame(columns=keys)
    for i in range(2,N+1):
        B_hat_Nth = B_hat[:i]
        sig = iisignature.sig(B_hat_Nth, order_model)
        sig = np.hstack((S, sig))
        temp_df = pd.DataFrame([sig], columns=keys)
        sig_df = pd.concat([temp_df, sig_df], ignore_index=True)

    sig_df = sig_df.iloc[::-1].reset_index(drop=True)

    return sig_df, keys

# Parameters
N = 1095
t_final, t_0 = 1, 0
mu = 0.01
sigma = 0.55
S_0 = 100
order_model = 4

# Simulate stock prices and Brownian motion
S, B, time = Simulate_Stock(S_0, mu, sigma, N, t_final, t_0)


# Prepare data for signature computation
B_hat = np.column_stack((time, B))

sig_df,keys = Signature(B_hat, N, order_model)


S = S[1:]
reg_sv_x=Lasso(alpha=.0001).fit(sig_df[keys], S)
pred_sv_x=reg_sv_x.predict(sig_df[keys])
print(len(time), len(pred_sv_x), len(S))

print('MSE of the traj with Extrapolated BMs ',mean_squared_error(S,pred_sv_x))

print(sig_df)


p = figure(title="Stock Price Simulation", x_axis_label='Time', y_axis_label='Stock Price',
           width=800, height=400)  # Correct width and height settings
p.line(time[1:], S, legend_label="Actual Stock Prices", line_color="tomato", line_width=1)
p.line(time[1:], pred_sv_x, legend_label="Predicted by Lasso", line_color="navy", line_width=1)
show(p)  # Open the plot in a browser


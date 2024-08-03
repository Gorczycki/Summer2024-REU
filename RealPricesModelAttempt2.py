import numpy as np
import matplotlib.pyplot as plt
import iisignature
import esig
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import pandas as pd
from bokeh.layouts import row
from bokeh.plotting import figure, show
import csv
from scipy.optimize import minimize



def read_GE():
    dr = pd.read_csv('GEreversed.csv')
    S = dr['close'].values

    return S

def log_returns(S):        
    r_t = np.log(S[1:] / S[:-1])  # Compute log returns

    return r_t

def QV_est_log(r_t):
    QV_est = []
    for k in range(len(r_t)):
        QV_est.append((r_t[k])**2)

    return QV_est

def get_rolling_vol_est(QV_est):
    window_size = 2
    est_daily_vol = []

    # Loop through the QV_est list, starting only after the first full window is available
    for k in range(window_size - 1, len(QV_est)):
        window_qv = QV_est[k-window_size+1:k+1]
        daily_vol = (sum(window_qv) / (window_size - 1))**0.5
        est_daily_vol.append(daily_vol)
    
    return est_daily_vol

def get_rolling_mu_est(r_t, est_vol_daily):
    rolling_mu = []
    window_size = 2
    for k in range(window_size - 1, len(r_t)):
        window_r = r_t[k-window_size+1:k+1]
        mu_hat = (np.sum(window_r) / len(window_r)) + (np.mean(est_vol_daily)**2 / 2)
        rolling_mu.append(mu_hat)
    return rolling_mu


def find_motion(r_t, est_vol_daily, window_size=11):
    delta_Wt = []
    
    # Adjust log returns to align with the windowed volatility estimates
    adjusted_r_t = r_t[window_size - 1:]
    
    for k in range(len(adjusted_r_t)):
        delta_Wt.append(adjusted_r_t[k] / est_vol_daily[k])
    
    return delta_Wt

def integrate_increments(W_t):
    B_t = np.cumsum(W_t)

    return B_t


def recover_stock_price(S0, est_mu_daily, est_vol_daily, B_sim, t):
    # Ensure lengths are consistent
    min_len = min(len(t), len(est_mu_daily), len(est_vol_daily), len(B_sim))
    t = t[:min_len]
    est_mu_daily = est_mu_daily[:min_len]
    est_vol_daily = est_vol_daily[:min_len]
    B_sim = B_sim[:min_len]

    # Compute the stock price for each time step
    S_t = S0 * np.exp((est_mu_daily - 0.5 * np.array(est_vol_daily)**2) * t + np.array(est_vol_daily) * B_sim)

    return S_t


"""
def mean_adjust(r_t):
    r_mean = np.mean(r_t)  # Compute the mean of log returns
    r_adjusted = r_t - r_mean  # Adjust the log returns
    
    return r_adjusted


def make_params(r_t):
    mu_hat = np.mean(r_t)
    sigma_squared_hat = np.var(r_t)

    return mu_hat, sigma_squared_hat



def recover_motion(mu_hat, sigma_squared_hat, days, t_0, t_final, S):
    time_grid = np.linspace(t_0, t_final, days)
    dt = time_grid[1] - time_grid[0]
    S_0 = S[0]
    dB = np.random.normal(0, np.sqrt(dt), days)
    #dB[0] = S_0
    #Consider not having dB[0] = S_0 at all in file
    B = np.cumsum(dB)
    sim_GBM = S_0 * np.exp((mu_hat - 0.5 * sigma_squared_hat) * time_grid + np.sqrt(sigma_squared_hat) * B)

    return sim_GBM, time_grid
"""
    


def main():
    t_0, t_final = 0, 1
    S = read_GE()
    S0 = S[0]
    days = len(S)
    r_t = log_returns(S)

    time = np.arange(len(S))  # Time index for stock prices
    time_adjusted = np.arange(len(r_t))  # Time index for adjusted log returns
    #mu_hat, sigma_squared_hat = make_params(r_t)
    #sim_GBM, time_grid = recover_motion(mu_hat, sigma_squared_hat, days, t_0, t_final, S)

    r_t = log_returns(S)
    QV_sim = QV_est_log(r_t)
    est_vol_daily = get_rolling_vol_est(QV_sim)
    est_mu_daily = get_rolling_mu_est(r_t, est_vol_daily)
    W_t = find_motion(r_t, est_vol_daily)
    B_sim = integrate_increments(W_t)
    t = np.linspace(t_0, t_final, len(B_sim))


    t_adjusted = np.arange(len(est_mu_daily))
    S_t = recover_stock_price(S0, est_mu_daily, est_vol_daily, B_sim, t_adjusted)

    print(S_t)




    """
    # Plot real stock price
    plt.subplot(3, 1, 1)
    plt.plot(time, S, label='Real Stock Price')
    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.title('Real Stock Price')
    plt.legend()

    # Plot recovered Brownian motion
    plt.subplot(3, 1, 2)
    plt.plot(np.arange(len(B_sim)), B_sim, label='Recovered Brownian Motion', color='green')
    plt.xlabel('Day')
    plt.ylabel('BM')
    plt.title('Recovered Brownian Motion')
    plt.legend()

    # Plot recovered stock price
    plt.subplot(3, 1, 3)
    plt.plot(t, S_t, label='Recovered Stock Price', color='orange')
    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.title('Recovered Stock Price')
    plt.legend()

    plt.tight_layout()
    plt.show()
    """


if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
import iisignature
import esig
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import pandas as pd
from bokeh.layouts import row
from bokeh.plotting import figure, show

def read_GE():
    S = []
    dr = pd.read_csv('GEreversed.csv')
    #dr = pd.read_csv('IBM_daily_close.csv')
    S = dr['close'].values
    return S

def get_QV_price(S, days):
    QV_hat_price = [np.sum(np.diff(S[:i+1]) ** 2) for i in range(1, days)]
    QV_hat_price = np.array(QV_hat_price).flatten()
    time_days = np.linspace(0, 1, days-1)  # Adjusted length

    r = figure(width=500, height=350, title='Quadratic variation (QV) estimation')
    r.line(time_days, QV_hat_price, legend_label='QV estimated of price', line_color='tomato')

    return QV_hat_price, time_days, r


def get_BM_price(S, QV_hat_price, time_days):
    sqrt_QV_hat = np.sqrt(QV_hat_price)
    new_increments_BM = [np.diff(S)[k] / sqrt_QV_hat[k] for k in range(len(sqrt_QV_hat)-1)]
    B_daily_ext = np.cumsum(new_increments_BM)
    B_daily_ext = np.insert(B_daily_ext, 0, 0)


    s = figure(width=500, height=350, title='Underlying Brownian motion')
    s.line(time_days, B_daily_ext, legend_label='Estimated BM', line_color='tomato')

    return B_daily_ext, s

"""
def Daily_log_returns(S, days):
    Daily_logs = []
    for k in range(days-1):
        Daily_logs.append(np.log(S[k+1]/S[k]))
    for k in range(len(Daily_logs)):
        Daily_logs[k] = np.abs(Daily_logs[k])
    Daily_logs = np.array(Daily_logs).flatten()

    return Daily_logs
"""

def Daily_log_returns(S, days):
    Daily_logs = []
    for k in range(days-1):
        Daily_logs.append(np.log(S[k+1]/S[k]))
    Daily_logs = np.abs(np.array(Daily_logs).flatten())  # Ensure all log returns are non-negative

    q = figure(width=500, height=350, title='volatility chart')
    q.line(days, Daily_logs, legend_label='volatility chart', line_color='pink')

    return Daily_logs, q


def compute_daily_volatility(log_returns):
    # Compute daily volatility as the standard deviation of log returns
    # Use a rolling window to compute volatility if you want a more advanced approach
    vol = np.std(log_returns)  # Standard deviation of log returns as a simple estimate
    return vol

def plot_daily_volatility(volatility):
    time_days = np.linspace(0, 1, len(volatility))
    vol_percentage = volatility * 100  # Convert to percentage

    p = figure(width=500, height=350, title='Daily Volatility as Percentage')
    p.line(time_days, vol_percentage, legend_label='Volatility (%)', line_color='pink')
    return p


def get_QV_vol(volatility, days):
    QV_hat_volatility = [np.sum(np.diff(volatility[:i+1]) ** 2) for i in range(1, days)]
    #QV_hat_volatility = [np.sum(np.diff(volatility[:i+1]) ** 2) for i in range(len(volatility))]
    QV_hat_volatility = np.array(QV_hat_volatility).flatten()
    time_days = np.linspace(0, 1, days-1)

    v = figure(width=500, height=350, title='Quadratic variation (QV) of volatility')
    v.line(time_days, QV_hat_volatility, legend_label='QV estimated of volatility', line_color='blue')

    return QV_hat_volatility, v


def get_BM_vol(volatility, QV_hat_volatility, days):
    # Ensure QV_hat_volatility does not contain negative values
    QV_hat_volatility = np.maximum(QV_hat_volatility, 1e-10)  # Small positive value to avoid division by zero
    sqrt_QV_hat_vol = np.sqrt(QV_hat_volatility)
    
    # Ensure that the increments are computed correctly and handle the case where sqrt_QV_hat_vol might be zero
    new_increments_BM_vol = [np.diff(volatility)[k] / sqrt_QV_hat_vol[k] if sqrt_QV_hat_vol[k] > 0 else 0 for k in range(len(sqrt_QV_hat_vol)-1)]
    BM_volatility_ext = np.cumsum(new_increments_BM_vol)
    BM_volatility_ext = np.insert(BM_volatility_ext, 0, 0)
    BM_volatility_ext = np.abs(BM_volatility_ext)
    time_days = np.linspace(0, 1, len(BM_volatility_ext))
    
    p = figure(width=500, height=350, title='Brownian motion of volatility')
    p.line(time_days, BM_volatility_ext, legend_label='Estimated BM of volatility', line_color='blue')

    return BM_volatility_ext, p

def Signature(B_hat, N, order_model):
    nbr_components = B_hat.shape[1]
    keys = esig.sigkeys(nbr_components, order_model).strip().split(" ")
    S = np.array([1])
    sig_df = pd.DataFrame(columns=keys)
    for i in range(1, N+1):
        B_hat_Nth = B_hat[:i]
        sig = iisignature.sig(B_hat_Nth, order_model)
        sig = np.hstack((S, sig))
        temp_df = pd.DataFrame([sig], columns=keys)
        sig_df = pd.concat([temp_df, sig_df], ignore_index=True)

    sig_df = sig_df.iloc[::-1].reset_index(drop=True)

    return sig_df, keys

def main():
    S = read_GE()
    days = len(S)
    S_0 = S[0]
    order_model = 2
    time_days = days


    S_vol, q = Daily_log_returns(S, days)
    QV_vol, v = get_QV_vol(S_vol, days)
    BM_vol, p = get_BM_vol(S_vol, QV_vol, days)
    print(BM_vol)
    QV_hat_price, time_days, r = get_QV_price(S, days)
    B_daily_ext, s = get_BM_price(S, QV_hat_price, time_days)
    Vol_percentage = compute_daily_volatility(S_vol)


    t_scaled=[k/len(B_daily_ext) for k in range(len(B_daily_ext))]
    #print("Length of volatility: ", len(S_vol))
    #print("Length of QV_hat_volatility: ", len(QV_vol))
    #print("Length of BM_volatility_ext: ", len(BM_vol))
    #print("Length of time_days: ", len(time_days))
    #print(len(QV_hat_price))
    #print(len(B_daily_ext))

    
    
    PP = np.column_stack((t_scaled, B_daily_ext, BM_vol)) #separate paths

    sig_df,keys = Signature(PP, days, order_model)

    print(sig_df)


    reg_sv_x=Lasso(alpha=.00001).fit(sig_df[keys], S)
    pred_sv_x=reg_sv_x.predict(sig_df[keys])

    check = reg_sv_x.coef_
    print(check)


    print('MSE of the traj with Extrapolated BMs ',mean_squared_error(S, pred_sv_x))
    

    # Create the figures
    vol_percentage_figure = plot_daily_volatility(S_vol)
    QV_vol_figure = v
    BM_vol_figure = p
    QV_price_figure = r
    BM_price_figure = s


    # Plot Lasso regression results
    lasso_figure = figure(title="Stock Price Simulation", x_axis_label='Time', y_axis_label='Stock Price', width=800, height=400)
    lasso_figure.line(t_scaled, S, legend_label="Actual Stock Prices", line_color="tomato", line_width=1)
    lasso_figure.line(t_scaled, pred_sv_x, legend_label="Predicted by Lasso", line_color="navy", line_width=1)

    # Arrange and show all plots
    BMCompare = row(BM_vol_figure, BM_price_figure)
    QVCompare = row(QV_vol_figure, QV_price_figure)
    L = row(vol_percentage_figure, BMCompare, QVCompare, lasso_figure)
    show(L)

    
    
    
        

if __name__ == "__main__":
    main()

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

#We see that the standard norm is shared between B and S
#We recover Brownian Motion by QV of price



"""
def BM(N, t_final, t_0):
    time_grid = np.linspace(t_0, t_final, num=N) #returns evenly spaced numbers over time interval, N ticks
    dt = time_grid[1] - time_grid[0]
    dB = np.random.normal(0, np.sqrt(dt), N) #randomly pulls from normal distr. with mean 0, variance of np.sqrt(dt), and N pulls 
    dB[0] = 0 #W_0 = 0
    B = np.cumsum(dB) #cumulative summation of np.random.normal
    return time_grid, B #returns cumulative summation of brownian motion in an array
"""


"""
def Simulate_Stock(S_0, mu, sigma, N, t_final, t_0):
    time_grid, B = BM(N, t_final, t_0) 
    S = S_0 * np.exp((mu - 0.5 * sigma**2) * time_grid + sigma * B) #Stock price with BM
    return S, B, time_grid
"""

def GEfiltered():
    #time_grid, B = BM(N, t_final, t_0)
    dr = pd.read_csv('GEreversed.csv')
    S = dr['close'].values
    B = np.diff(np.log(S))
    B = np.insert(B, 0, 0).cumsum()

    return S, B #, time_grid


def get_QV_price_Heston(S, B, sigma, days):
    QV_hat_price = []
    for k in range(days - 1):  # Loop over the range to avoid index errors
        QV_hat_price.append((S[k+1] - S[k]) ** 2)  # Calculate QV using daily data points correctly
    QV_hat_price = np.array(QV_hat_price).flatten()

    S_daily = S  # S is already daily
    B_daily = B[:days]  # Extract daily data points from B

    QV_real_price = sigma ** 2 * (S_daily[:-1]) ** 2  # Exclude last element to match QV_hat_price length
    time_days = np.array(range(days - 1))  # Adjust length to match QV arrays
    time_days = time_days / len(time_days)

    r = figure(width=500, height=350, title='Quadratic variation (QV) estimation')
    r.line(time_days, QV_real_price, legend_label='QV real of price', line_color='royalblue')
    r.line(time_days, QV_hat_price, legend_label='QV estimated of price', line_color='tomato')

    return QV_hat_price, QV_real_price, S_daily, B_daily, time_days, r




def get_BM_price(S_daily, B_daily, QV_hat_price, time_days):
    sqrt_QV_hat = np.sqrt(QV_hat_price)  # Takes square root of quadratic variation price
    new_increments_BM = [np.diff(S_daily)[k] / sqrt_QV_hat[k] for k in range(len(sqrt_QV_hat)) if sqrt_QV_hat[k] != 0]
    B_daily_ext = np.cumsum(new_increments_BM)
    B_daily_ext = np.insert(B_daily_ext, 0, 0)


    s = figure(width=500, height=350, title='Compare Brownian motions')
    s.line(time_days, B_daily[:-1], legend_label='Real BM', line_color='royalblue')  # Adjust length to match time_days
    s.line(time_days, B_daily_ext, legend_label='Estimated BM', line_color='tomato')

    return B_daily[:-1], B_daily_ext, s

def Signature(B_hat, N, order_model):
    nbr_components = B_hat.shape[1]  # Shape of 2d array, number of components is time, returns 2
    keys = esig.sigkeys(nbr_components, order_model).strip().split(" ")
    S = np.array([1])
    sig_df = pd.DataFrame(columns=keys)

    for i in range(1, N + 1):
        B_hat_Nth = B_hat[:i]  # Signatures taken at each time step
        sig = iisignature.sig(B_hat_Nth, order_model)  # Signature with level
        sig = np.hstack((S, sig))  # Takes this signature alongside S price
        temp_df = pd.DataFrame([sig], columns=keys)  # Puts into dataframe
        sig_df = pd.concat([temp_df, sig_df], ignore_index=True)  # Matches the two

    sig_df = sig_df.iloc[::-1].reset_index(drop=True)  # Integer position-based indexing

    # Check for inf and NaN values in sig_df
    if sig_df.isin([np.inf, -np.inf]).any().any():
        raise ValueError("Infinite values found in signature dataframe")
    if sig_df.isna().any().any():
        raise ValueError("NaN values found in signature dataframe")

    return sig_df, keys



def main():

    #hours=8
    #minutes=60*5
    #days=1095
    df = pd.read_csv('GEfiltered.csv')
    days = len(df)
    t_final, t_0 = 1, 0

    sigma = 0.2

    order_model = 2


    #time_grid = np.linspace(t_0, t_final, num=len(S))  # Create a time grid matching the number of data points

    S, B = GEfiltered()
    #QV_hat_price, QV_real_price, S_daily, B_daily, time_days, r = get_QV_price_Heston(S, B, sigma, days, hours, minutes)
    QV_hat_price, QV_real_price, S_daily, B_daily, time_days, r = get_QV_price_Heston(S, B, sigma, days)
    B_daily, B_daily_ext, s = get_BM_price(S_daily, B_daily, QV_hat_price, time_days)

    t_scaled=[k/len(B_daily_ext) for k in range(len(B_daily_ext))] #???
    B_hat = np.column_stack((t_scaled, B_daily_ext)) #stacks the 1d arrays of t and B into 2d array as columns
    sig_df, keys = Signature(B_hat, days, order_model)

    #print(QV_real_price)
    #print(QV_hat_price)


    #print(len(B_daily_ext))
    #print(len(t_scaled)) 
    #print(len(S_daily))

    #print(t_scaled) 
    #b_dub = B_hat.shape[1] #returns 2 when printed 

    #print(keys) 
    #print(sig_df) 

    #print(QV_hat_price) 
    print(sig_df)

    #regressing: X is the signature values, Y is daily price
    reg_sv_x=Lasso(alpha=.00001).fit(sig_df[keys], S_daily) #takes the keys of the signature alongside daily price
    pred_sv_x=reg_sv_x.predict(sig_df[keys])

    print('MSE of the traj with Extrapolated BMs ',mean_squared_error(S_daily,pred_sv_x))
    print(len(sig_df[keys]))
    print(len(S_daily))


    p = figure(title="Stock Price Simulation", x_axis_label='Time', y_axis_label='Stock Price',
           width=800, height=400)  # Correct width and height settings
    p.line(t_scaled, S_daily, legend_label="Actual Stock Prices", line_color="tomato", line_width=1)
    p.line(t_scaled, pred_sv_x, legend_label="Predicted by Lasso", line_color="navy", line_width=1)
    L = row(s,r,p)
    show(L)

if __name__ == "__main__":
    main()

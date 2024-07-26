import numpy as np
import matplotlib.pyplot as plt
import iisignature
import esig
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import pandas as pd
from bokeh.layouts import row
from bokeh.plotting import figure, show

def BM(N, t_final, t_0):
    time_grid = np.linspace(t_0, t_final, num=N)
    dt = time_grid[1] - time_grid[0]
    dB = np.random.normal(0, np.sqrt(dt), N)
    dB[0] = 0
    B = np.cumsum(dB)
    return time_grid, B

def Simulate_Stock(S_0, mu, sigma, N, t_final, t_0):
    time_grid, B = BM(N, t_final, t_0)
    S = S_0 * np.exp((mu - 0.5 * sigma**2) * time_grid + sigma * B)
    return S, B, time_grid

def Signature(B_hat, N, order_model):
    nbr_components = B_hat.shape[1]
    keys=esig.sigkeys(nbr_components,order_model).strip().split(" ")
    S = np.array([1])
    sig_df = pd.DataFrame(columns=keys)
    for i in range(1,N+1):
        B_hat_Nth = B_hat[:i]
        sig = iisignature.sig(B_hat_Nth, order_model)
        sig = np.hstack((S, sig))
        temp_df = pd.DataFrame([sig], columns=keys)
        sig_df = pd.concat([temp_df, sig_df], ignore_index=True)

    sig_df = sig_df.iloc[::-1].reset_index(drop=True)

    return sig_df, keys


def get_QV_price_Heston(S,B,sigma,days,hours,minutes):
    QV_hat_price=[]
    for k in range(days):
        QV_hat_price.append(days*np.sum(np.diff(S[k*(hours*minutes):(k+1)*(hours*minutes)])**2))
    QV_hat_price=np.array(QV_hat_price).flatten()

    S_daily=np.array([S[hours*minutes*k] for k in range(days)])
    B_daily=np.array([B[hours*minutes*k] for k in range(days)])

    QV_real_price=(sigma**2)*(S_daily)**2
    time_days=np.array(range(days))
    time_days=time_days/len(time_days)

    r = figure(width=500, height=350,title='Quadratic variation (QV) estimation')
    r1 = r.line(time_days,QV_real_price,legend_label='QV real of price', line_color='royalblue')
    r2 = r.line(time_days,QV_hat_price,legend_label='QV estimated of price',line_color='tomato')

    return QV_hat_price, QV_real_price, S_daily, B_daily, time_days, r

def get_BM_price(X_daily, B_daily, QV_hat_price,time_days):
    sqrt_QV_hat=np.sqrt(QV_hat_price)
    new_increments_BM=[np.diff(X_daily)[k]/sqrt_QV_hat[k] for k in range(len(sqrt_QV_hat)-1)]
    B_daily_ext=np.cumsum(new_increments_BM)
    B_daily_ext=np.insert(B_daily_ext,0,0)

    s = figure(width=500, height=350,title='Compare Brownian motions')
    s1 = s.line(time_days,B_daily, legend_label='Real BM', line_color='royalblue')
    s2 = s.line(time_days,B_daily_ext,legend_label='Estimated BM',line_color='tomato')

    return B_daily, B_daily_ext,s


hours=8
minutes=60*5
days=1095
N=hours * days * minutes
t_final, t_0 = 1, 0

mu = 0.01
sigma = .2
S_0 = 1

order_model = 3


S, B, time_grid = Simulate_Stock(S_0, mu, sigma, N, t_final, t_0)
QV_hat_price, QV_real_price, S_daily, B_daily, time_days, r = get_QV_price_Heston(S,B,sigma,days,hours,minutes)
B_daily, B_daily_ext,s = get_BM_price(S_daily, B_daily, QV_hat_price,time_days)

print(QV_real_price)
print(QV_hat_price)


print(len(B_daily_ext))
t_scaled=[k/len(B_daily_ext) for k in range(len(B_daily_ext))]
B_hat = np.column_stack((t_scaled, B_daily_ext))
print(len(t_scaled))
print(len(S_daily))

sig_df,keys = Signature(B_hat, days, order_model)
print(sig_df)


reg_sv_x=Lasso(alpha=.00001).fit(sig_df[keys], S_daily)
pred_sv_x=reg_sv_x.predict(sig_df[keys])

print('MSE of the traj with Extrapolated BMs ',mean_squared_error(S_daily,pred_sv_x))


p = figure(title="Stock Price Simulation", x_axis_label='Time', y_axis_label='Stock Price',
           width=800, height=400)  # Correct width and height settings
p.line(t_scaled, S_daily, legend_label="Actual Stock Prices", line_color="tomato", line_width=1)
p.line(t_scaled, pred_sv_x, legend_label="Predicted by Lasso", line_color="navy", line_width=1)
L = row(s,r,p)
show(L)


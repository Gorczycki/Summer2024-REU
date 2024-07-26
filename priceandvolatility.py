import numpy as np
import matplotlib.pyplot as plt
import iisignature
import esig
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import pandas as pd
from bokeh.layouts import row
from bokeh.plotting import figure, show, column


def BM(N, rho, t_final, t_0):
    time_grid = np.linspace(t_0, t_final, num=N)
    dt = time_grid[1] - time_grid[0]
    dB = np.random.normal(0, np.sqrt(dt), N)
    dB[0] = 0
    dW = rho*dB+np.sqrt(1-(rho)**2)*np.random.normal(0,np.sqrt(dt),N)
    dW[0]=0
    B = np.cumsum(dB)
    W = np.cumsum(dW)
    return time_grid, B, W


def Simulation(S_0, V_0, mu, N, rho, kappa, theta, alpha, t_final, t_0):
    time_grid, B, W = BM(N,rho, t_final, t_0)
    V = np.zeros(N)
    V[0] = V_0
    S = np.zeros(N)
    S[0] = S_0
    for k in range(N-1):
        #Reference 4.5 pg 27
        V[k+1]=V[k]+kappa*(theta-V[k])*(1/N)+(alpha)*(np.sqrt(V[k]))*(W[k+1]-W[k])
        S[k+1]=S[k]+(S[k])*(mu)*(1/N)+(np.sqrt(V[k]))*(S[k])*(B[k+1]-B[k])

    return  S,V,B,W,time_grid

def augmentTime(N, days, hours, minutes, S, V, B, W, time_grid):
    S_daily = np.array([S[hours*minutes*k] for k in range(days)])
    V_daily = np.array([V[hours*minutes*k] for k in range(days)])
    B_daily = np.array([B[hours*minutes*k] for k in range(days)])
    W_daily = np.array([W[hours*minutes*k] for k in range(days)])
    time_grid_daily = np.array([time_grid[hours*minutes*k] for k in range(days)])
    return S_daily, V_daily, B_daily, W_daily, time_grid_daily



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

def QVvol(V, V_daily,alpha,days, hours, minutes, time_grid_daily):
    QV_hat_vol=[]
    for k in range(days):
        #Reference 4.3 pg 25
        QV_hat_vol.append(days*np.sum(np.diff(V[k*(hours*minutes):(k+1)*(hours*minutes)])**2))
    QV_hat_vol=np.array(QV_hat_vol).flatten()
    QV_real_vol=(alpha**2)*(V_daily)

    p = figure(width=500, height=350,title='Quadratic variation Volatility')
    p1 = p.line(time_grid_daily,QV_real_vol, legend_label='QV real of Vol', line_color='royalblue')
    p2 = p.line(time_grid_daily,QV_hat_vol,legend_label='QV estimated of Vol',line_color='tomato')

    return QV_hat_vol, QV_real_vol, p

def BMvol(V_daily, W_daily, QV_hat_vol ,time_grid_daily):
    # Reference 4.3 pg 25
    sqrt_QV_hat=np.sqrt(QV_hat_vol)
    new_increments_BM=[np.diff(V_daily)[k]/sqrt_QV_hat[k] for k in range(len(sqrt_QV_hat)-1)]

    W_daily_ext=np.cumsum(new_increments_BM)
    W_daily_ext=np.insert(W_daily_ext,0,0)

    q = figure(width=500, height=350,title='Compare Brownian motions for Volatility')
    q1 = q.line(time_grid_daily,W_daily, legend_label='Real BM', line_color='royalblue')
    q2 = q.line(time_grid_daily,W_daily_ext,legend_label='Estimated BM',line_color='tomato')

    return W_daily, W_daily_ext,q

def QVprice(S,S_daily,V_daily,days, hours, minutes, time_grid_daily):
    QV_hat_price=[]
    #Calculates quadratic variation between each time step in "days" intervals of 3
    # Reference 4.3 pg 25
    for k in range(days):
        QV_hat_price.append(days*np.sum(np.diff(S[k*(hours*minutes):(k+1)*(hours*minutes)])**2))
    QV_hat_price=np.array(QV_hat_price).flatten()
    QV_real_price=V_daily*(S_daily)**2


    r = figure(width=500, height=350,title='Quadratic variation Price')
    r1 = r.line(time_grid_daily,QV_real_price,legend_label='QV real of price', line_color='royalblue')
    r2 = r.line(time_grid_daily,QV_hat_price,legend_label='QV estimated of price',line_color='tomato')

    return QV_hat_price, QV_real_price, r

def BMprice(S_daily, B_daily, QV_hat_price,time_grid_daily):
    # Reference 4.3 pg 25
    sqrt_QV_hat=np.sqrt(QV_hat_price)
    new_increments_BM=[np.diff(S_daily)[k]/sqrt_QV_hat[k] for k in range(len(sqrt_QV_hat)-1)]

    B_daily_ext=np.cumsum(new_increments_BM)
    B_daily_ext=np.insert(B_daily_ext,0,0)

    s = figure(width=500, height=350,title='Compare Brownian motion for Price')
    s1 = s.line(time_grid_daily,B_daily, legend_label='Real BM', line_color='royalblue')
    s2 = s.line(time_grid_daily,B_daily_ext,legend_label='Estimated BM',line_color='tomato')

    return B_daily, B_daily_ext,s





hours=8
minutes=60*5
days=1095
N=hours * days * minutes
t_final, t_0 = 1, 0

mu = 0.01
sigma = .15
alpha=0.25
theta=0.15
kappa=0.5
rho=-0.5

S_0 = 1
V_0 = .08
order_model = 2


S, V, B, W, time_grid = Simulation(S_0, V_0, mu, N, rho,kappa, theta, alpha, t_final, t_0)
S_daily, V_daily, B_daily, W_daily, time_grid_daily = augmentTime(N, days, hours, minutes, S, V, B, W, time_grid)

QV_hat_vol, QV_real_vol, p = QVvol(V,V_daily, alpha,days, hours, minutes, time_grid_daily)
W_daily, W_daily_ext, q = BMvol(V_daily, W_daily, QV_hat_vol, time_grid_daily)

QV_hat_price, QV_real_price, r = QVprice(S, S_daily, V_daily, days, hours, minutes, time_grid_daily)
B_daily, B_daily_ext,s = BMprice(S_daily, B_daily, QV_hat_price,time_grid_daily)


t_scaled=[k/len(B_daily_ext) for k in range(len(B_daily_ext))]

PP = np.column_stack((t_scaled, B_daily_ext, W_daily_ext))

sig_df,keys = Signature(PP, days, order_model)

print(sig_df)


reg_sv_x=Lasso(alpha=.00001).fit(sig_df[keys], S_daily)
pred_sv_x=reg_sv_x.predict(sig_df[keys])

check = reg_sv_x.coef_
print(check)


print('MSE of the traj with Extrapolated BMs ',mean_squared_error(S_daily, pred_sv_x))


c = figure(title="Stock Price Simulation", x_axis_label='Time', y_axis_label='Stock Price', width=800, height=400)
c.line(t_scaled, S_daily, legend_label="Actual Stock Prices", line_color="tomato", line_width=1)
c.line(t_scaled, pred_sv_x, legend_label="Predicted by Lasso", line_color="navy", line_width=1)

BMCompare = row(q,s)
QVCompare = row(p,r)
real = column(BMCompare,QVCompare, c)
show(real)
#L = row(s,r,p)
#show(L)

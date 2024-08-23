
#Call packages
import itertools as itt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
from matplotlib.pyplot import cm
import math
from functools import partial
import time as tm
from scipy.integrate import quad
import cmath
import numpy as np
import matplotlib.pyplot as plt
from math import isnan
from scipy.optimize import bisect, least_squares, minimize_scalar, minimize
import os
from py_vollib.ref_python.black_scholes_merton.implied_volatility import implied_volatility
from py_vollib.ref_python.black_scholes_merton.implied_volatility import black_scholes_merton
from scipy.stats import norm


def duplicate(testList, n):
    x = [list(testList) for _ in range(n)]
    flat_list = []
    for sublist in x:
        for item in sublist:
            flat_list.append(item)
    return flat_list


def Heston_P_Value(hestonParams, r, T, s0, K, typ):
    kappa, theta, sigma, rho, v0 = hestonParams
    return 0.5 + (1. / np.pi) * \
        quad(lambda xi: Int_Function_1(xi, kappa, theta, sigma, rho, v0, r, T, s0, K, typ), 0., 500.)[0]


def Int_Function_1(xi, kappa, theta, sigma, rho, v0, r, T, s0, K, typ):
    return (cmath.e ** (-1j * xi * np.log(K)) * Int_Function_2(xi, kappa, theta, sigma, rho, v0, r, T, s0, typ) / (
                1j * xi)).real


def Int_Function_2(xi, kappa, theta, sigma, rho, v0, r, T, s0, typ):
    if typ == 1:
        w = 1.
        b = kappa - rho * sigma
    else:
        w = -1.
        b = kappa
    ixi = 1j * xi
    d = cmath.sqrt((rho * sigma * ixi - b) * (rho * sigma * ixi - b) - sigma * sigma * (w * ixi - xi * xi))
    g = (b - rho * sigma * ixi - d) / (b - rho * sigma * ixi + d)
    ee = cmath.e ** (-d * T)
    C = r * ixi * T + kappa * theta / (sigma * sigma) * (
                (b - rho * sigma * ixi - d) * T - 2. * cmath.log((1.0 - g * ee) / (1. - g)))
    D = ((b - rho * sigma * ixi - d) / (sigma * sigma)) * (1. - ee) / (1. - g * ee)
    return cmath.e ** (C + D * v0 + ixi * np.log(s0))


def phi(x):  ## Gaussian density
    return np.exp(-x * x / 2.) / np.sqrt(2 * np.pi)


#### Black Sholes Vega
def BlackScholesVegaCore(DF, F, X, T, v):  # S=F*DF
    vsqrt = v * np.sqrt(T)
    d1 = (np.log(F / X) + (vsqrt * vsqrt / 2.)) / vsqrt
    return F * phi(d1) * np.sqrt(T) / DF


#### Black Sholes Function
def BlackScholesCore(CallPutFlag, DF, F, X, T, v):
    ## DF: discount factor
    ## F: Forward
    ## X: strike
    vsqrt = v * np.sqrt(T)
    d1 = (np.log(F / X) + (vsqrt * vsqrt / 2.)) / vsqrt
    d2 = d1 - vsqrt
    if CallPutFlag:
        return DF * (F * norm.cdf(d1) - X * norm.cdf(d2))
    else:
        return DF * (X * norm.cdf(-d2) - F * norm.cdf(-d1))


##  Black-Scholes Pricing Function
def BlackScholes(CallPutFlag, S, X, T, r, d, v):
    ## r, d: continuous interest rate and dividend
    return BlackScholesCore(CallPutFlag, np.exp(-r * T), np.exp((r - d) * T) * S, X, T, v)


def heston_EuropeanCall(hestonParams, r, T, s0, K):
    a = s0 * Heston_P_Value(hestonParams, r, T, s0, K, 1)
    b = K * np.exp(-r * T) * Heston_P_Value(hestonParams, r, T, s0, K, 2)
    return a - b


def heston_Vanilla(hestonParams, r, T, s0, K, flag):
    a_call = s0 * Heston_P_Value(hestonParams, r, T, s0, K, 1)
    b_call = K * np.exp(-r * T) * Heston_P_Value(hestonParams, r, T, s0, K, 2)
    a_put = s0 * (1 - Heston_P_Value(hestonParams, r, T, s0, K, 1))
    b_put = K * np.exp(-r * T) * (1 - Heston_P_Value(hestonParams, r, T, s0, K, 2))
    if flag == 'call':
        return a_call - b_call
    if flag == 'put':
        return b_put - a_put
    else:
        return print('You have chosen a flag which is not a Vanilla Option')


def heston_Impliedvol(hestonParams, r, T, s0, K):
    myPrice = heston_EuropeanCall(hestonParams, r, T, s0, K)

    ## Bisection algorithm when the Lee-Li algorithm breaks down
    def smileMin(vol, *args):
        K, s0, T, r, price = args
        return price - BlackScholes(True, s0, K, T, r, 0., vol)

    vMin = 0.000001
    vMax = 10.
    return bisect(smileMin, vMin, vMax, args=(K, s0, T, r, myPrice), rtol=1e-15, full_output=False, disp=True)


def phi(x):  ## Gaussian density
    return np.exp(-x * x / 2.) / np.sqrt(2 * np.pi)


#### Black Sholes Vega
def BlackScholesVegaCore(DF, F, X, T, v):  # S=F*DF
    vsqrt = v * np.sqrt(T)
    d1 = (np.log(F / X) + (vsqrt * vsqrt / 2.)) / vsqrt
    return F * phi(d1) * np.sqrt(T) / DF


def implied_vol_minimize(price, S0, K, T, r, payoff="call", disp=True):
    """ Returns Implied volatility by minimization"""

    n = 2  # must be even

    def obj_fun(vol):
        return (BlackScholes(True, S0, K, T, r, 0., vol) - price) ** n

    res = minimize_scalar(obj_fun, bounds=(1e-15, 8), method='bounded')
    if res.success == True:
        return res.x
    if disp == True:
        print("Strike", K)
    return -1

def get_vegas(maturities, strikes, initial_price, iv_market, flag_truncation):
    'Compute vega weights'
    vega=[]
    for i in range(len(maturities)):
        for j in range(len(strikes[i])):
            if flag_truncation==True:
                vega.append(min(1/(BlackScholesVegaCore(1,initial_price,strikes[i][j],maturities[i],iv_market[i,j])),1))
            else:
                vega.append(1/(BlackScholesVegaCore(1,initial_price,strikes[i][j],maturities[i],iv_market[i,j])))
    vega=np.array(vega)

    vega_by_mat=np.array(np.split(vega,len(maturities)))
    sums_each_strike=np.sum(vega_by_mat, axis=1)
    normalized_vega=np.array([vega_by_mat[j]/sums_each_strike[j] for j in range(len(maturities))])
    flat_normal_weights=normalized_vega.flatten()
    return flat_normal_weights, normalized_vega



maturities=np.array([0.10,.25, 0.3, 0.5 ,.75, 1.,2])
strikes=[]
nbr_strikes=7
S0=1


def get_iv_Heston_by_params(set_params, strikes, maturities, S0, nbr_strikes):
    'This function generates for the chosen parameters under Q of the Heston, the strikes and the maturities; the IV surface'
    r = 0.
    index_selected_mat = range(len(maturities))
    print(index_selected_mat)
    hestonParams = set_params['kappa'], set_params['theta'], set_params['alpha'], set_params['rho'], set_params['v0']

    Heston_prices_calib_call = []
    iv_call = []
    for j in index_selected_mat:
        print('----', j)
        for strike in strikes[j]:
            he_p_call = heston_Vanilla(hestonParams, r, maturities[j], S0, strike, 'call')
            Heston_prices_calib_call.append(he_p_call)
            print("test:", he_p_call, S0, strike, maturities[j])
            iv_call.append(implied_volatility(he_p_call, S0, strike, maturities[j], 0, 0, 'c'))
            print(iv_call)


    Heston_prices_calib_put = []
    iv_put = []
    for j in index_selected_mat:
        for strike in strikes[j]:
            he_p_put = heston_Vanilla(hestonParams, r, maturities[j], S0, strike, 'put')
            Heston_prices_calib_put.append(he_p_put)
            iv_put.append(implied_volatility(he_p_put, S0, strike, maturities[j], 0, 0, 'p'))

    element_to_substitute = 0

    Heston_prices_calib = []

    for j in index_selected_mat:
        for k in range(len(strikes[j])):
            if (k < len(strikes[0]) - element_to_substitute):
                Heston_prices_calib.append(heston_Vanilla(hestonParams, r, maturities[j], S0, strikes[j][k], 'call'))
            else:
                Heston_prices_calib.append(heston_Vanilla(hestonParams, r, maturities[j], S0, strikes[j][k], 'put'))

    print(len(Heston_prices_calib))
    fig1 = plt.figure(figsize=(12, 8))
    print(iv_call[0])
    print(len(strikes))

    for j in range(len(maturities)):
        #plt.subplot(2, 2, j+1)
        plt.plot(strikes[j], iv_call[j * nbr_strikes:(j+1) * nbr_strikes],
                 label='Maturity T={}'.format(round(maturities[j], 4)), marker='o')
        plt.legend()
    plt.title('Generated Smiles')
    plt.show()

    np.array(np.split(np.array(iv_call), len(maturities)))


    # set up a figure twice as wide as it is tall
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    YY = np.array([[maturities[j]] * (len(strikes[0])) for j in range(7)])
    ax.plot_surface(strikes, YY, np.array(np.split(np.array(iv_call), len(maturities))), rstride=1, cstride=1,
                    cmap=cm.viridis,
                    linewidth=0.5)

    ax.set_xlabel('Strikes')
    ax.set_ylabel('Maturities')
    ax.set_zlabel('IV')
    ax.set_title('Implied Volatililty');

    plt.tight_layout()
    plt.show()

    return np.array(Heston_prices_calib), np.array(iv_call)

strikes_new=np.array([np.array([0.8,0.85,0.9,0.925,0.95,0.975,1,1.025,1.05,1.075,1.1,1.15,1.2]) for k in range(7)])
set_params={'alpha':0.4,'kappa':0.1,'theta':0.1,'rho':-0.5,'v0':0.08}
prices, iv= get_iv_Heston_by_params(set_params,strikes_new,maturities,S0,len(strikes_new[0]))
iv_market=np.array(np.split(iv,len(maturities)))
prices_market=np.array(np.split(prices,len(maturities)))
print(strikes_new)
print(prices)
print(iv)

np.save("strikes_sim",strikes_new)
np.save("optionprice_sim",prices_market)
np.save("iv_sim",iv_market)

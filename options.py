import pandas as pd
import numpy as np
import os
from tqdm.auto import tqdm
import iisignature
import torch
from math import factorial
from joblib import Parallel,delayed
import esig
import itertools as itt
from scipy.optimize import least_squares,minimize
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from matplotlib import cm


def implied_vol_minimize(price, S0, K, T, r, payoff="call", disp=True):
    """ Returns Implied volatility by minimization"""

    n = 2  # must be even
    print(T)
    T = T/365
    print("TEST",T)
    def obj_fun(vol):
        return (BlackScholes(True, S0, K, T, r, 0., vol) - price) ** n

    res = minimize_scalar(obj_fun, bounds=(1e-15, 8), method='bounded')
    if res.success == True:
        return res.x
    if disp == True:
        print("Strike", K)
    return -1


def duplicate(testList, n):
    x = [list(testList) for _ in range(n)]
    flat_list = []
    for sublist in x:
        for item in sublist:
            flat_list.append(item)
    return flat_list


def multi_maturities(maturities, k):
    mat = list(maturities)
    new_multi_mat = []
    for element in mat:
        for j in range(k):
            new_multi_mat.append(element)
    return np.array(new_multi_mat)


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
        return DF * (F *norm.cdf(d1) - X * norm.cdf(d2))
    else:
        return DF * (X * norm.cdf(-d2) - F * norm.cdf(-d1))


##  Black-Scholes Pricing Function
def BlackScholes(CallPutFlag, S, X, T, r, d, v):
    ## r, d: continuous interest rate and dividend
    return BlackScholesCore(CallPutFlag, np.exp(-r * T), np.exp((r - d) * T) * S, X, T, v)


def tilde_transformation(word):
    word_aux = word.copy()
    word_aux_2 = word.copy()
    if word[-1] == 1:
        word.append(2)
        word_aux.append(3)
        return word, word_aux
    if word[-1] != 1:
        word_aux.append(2)
        word_aux_2.append(3)
        word[-1] = 1
        return word_aux, word_aux_2, word


def e_tilde_part2_new(words_as_lists):
    tilde = [list(tilde_transformation(words_as_lists[k])) for k in
             np.array(range(len(words_as_lists)))[1:]]  # we skip the empty word
    return tilde


def from_tilde_to_strings_new(tilde):
    for k in range(len(tilde)):
        if len(tilde[k]) == 2:
            tilde[k][0] = str(tuple(tilde[k][0])).replace(" ", "")
            tilde[k][1] = str(tuple(tilde[k][1])).replace(" ", "")
        elif (len(tilde[k]) == 3 and len(tilde[k][-1]) == 1):
            tilde[k][0] = str(tuple(tilde[k][0])).replace(" ", "")
            tilde[k][1] = str(tuple(tilde[k][1])).replace(" ", "")
            tilde[k][-1] = str(tuple(tilde[k][-1])).replace(",", "")
        elif len(tilde[k]) == 3:
            tilde[k][0] = str(tuple(tilde[k][0])).replace(" ", "")
            tilde[k][1] = str(tuple(tilde[k][1])).replace(" ", "")
            tilde[k][2] = str(tuple(tilde[k][2])).replace(" ", "")
    return tilde


def get_tilde_df_debug(Sig_data_frame, new_tilde, keys_n, keys_n1, comp_of_path, rho, y):
    aus_B = []
    y = [[eval(key)] if isinstance(eval(key), int) else list(eval(key)) for key in keys_n]

    for k in range(len(y)):
        if k == 0:
            aus_B.insert(0, Sig_data_frame['(2)'])
        if (k > 0 and y[k][-1] == 1):
            aus_B.append(Sig_data_frame[new_tilde[k - 1][0]])

        if (k > 0 and y[k][-1] == 2):
            aus_B.append(Sig_data_frame[new_tilde[k - 1][0]] - 0.5 * Sig_data_frame[new_tilde[k - 1][2]])

        if (k > 0 and y[k][-1] == 3):
            aus_B.append(Sig_data_frame[new_tilde[k - 1][0]] - rho * 0.5 * Sig_data_frame[new_tilde[k - 1][2]])

    new_keys_B = [keys_n1[k] + str('~B') for k in range(len(y))]
    new_dictionary_B = {key: series for key, series in zip(new_keys_B, aus_B)}
    transformed_data_frame_B = pd.DataFrame(new_dictionary_B)
    return transformed_data_frame_B


def col_stack(element):
    return np.column_stack((element[0], element[1], element[2]))


def correlated_bms_correct(N, rho, t_final, t_0):
    time_grid = np.linspace(0, t_final, num=N, retstep=True)[0]
    dt = np.abs(time_grid[0] - time_grid[1])
    dB = np.random.normal(0, np.sqrt(dt), N)
    dB[0] = 0
    dW = rho * dB + np.sqrt(1 - (rho) ** 2) * np.random.normal(0, np.sqrt(dt), N)
    dW[0] = 0
    B = np.cumsum(dB)
    W = np.cumsum(dW)
    return time_grid, B, W

"[31.04, 60.04, 93.04, 172.00, 354, 599.00, 781.04]"
maturities = np.array([30,60,99,179,361,515,879])
strikes = np.array([3300,3500,3700,3900,4100,4300,4500,4700,4900])
#strikes = np.array([.8,.85,.9,.95,1,1.05,1.1,1.15,1.2])
iv_market=np.load('iv_data.npy')
market_prices=np.load('optionprice.npy')
option_prices_splitted=np.array(np.split(market_prices,len(maturities)))

prices_scaled=option_prices_splitted



fig = plt.figure(figsize=(15,8))
ax = plt.axes(projection='3d')

X, Y = np.meshgrid(strikes, maturities)

fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot using the mesh grids and the corresponding IV data
# Note: X, Y, and iv_market must align in shape
ax.plot_surface(X, Y, iv_market, rstride=1, cstride=1, cmap=cm.winter, edgecolor='none')

ax.set_xlabel('Strike Prices')
ax.set_ylabel('Maturities (Days to Expiry)')
ax.set_zlabel('Implied Volatility (IV)')
ax.set_title('Implied Volatility Surface for SPX Options')





def get_vegas(maturities, strikes, initial_price, iv_market, flag_truncation):
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

initial_price=4554.96
strikes_all=duplicate(strikes,len(maturities))
strikes=np.array([np.array(strikes_all[j*(len(strikes)):(j+1)*(len(strikes))]) for j in range(len(maturities))])
maturities_ext=multi_maturities(maturities,len(strikes[0]))
strike_flat=strikes.flatten()
option_prices_splitted=np.array(np.split(market_prices,len(maturities)))

flag_truncation=False #Truncation was introduced by Cont and Ben Hamida (2011)
flat_normal_weights, norm_vegas=get_vegas(maturities, strikes, initial_price, iv_market, flag_truncation)

MC_number, rho, n= 100, -0.5, 2
rho_matrix=[[1,rho],[rho,1]]
d,D=2,1
grid=np.array([int(T) for T in maturities])
maturity_dict=dict(zip(maturities, grid.T))

nbr_components=len(rho_matrix)+1
keys_n1=esig.sigkeys(nbr_components,n+1).strip().split(" ")
keys_n=esig.sigkeys(nbr_components,n).strip().split(" ")
y=[[eval(key)] if isinstance(eval(key), int) else list(eval(key)) for key in keys_n]
first_step=e_tilde_part2_new(y)
new_tilde=from_tilde_to_strings_new(first_step)



def auxiliar_function(idx_mat,n,nbr_components,keys_n1,keys_n,new_tilde,maturity_dict):
    time_and_bms=correlated_bms_correct(maturity_dict[maturities[idx_mat]],rho,maturities[idx_mat],0)
    augmented_bms=col_stack(time_and_bms)
    S = np.array([1])
    sig_df = pd.DataFrame(columns=keys_n)
    sig_df.loc[0] = [0] * len(keys_n)
    for i in range(1,maturities[idx_mat]):
        augmented_bms_run = augmented_bms[:i]
        sig = iisignature.sig(augmented_bms_run, n+1)
        sig = np.hstack((S, sig))
        temp_df = pd.DataFrame([sig], columns=keys_n1)
        sig_df = pd.concat([temp_df, sig_df], ignore_index=True)

    sig_df = sig_df.iloc[::-1].reset_index(drop=True)

    tilde_sig_df=get_tilde_df_debug(sig_df,new_tilde,keys_n,keys_n1,nbr_components,rho,y)
    tilde_sig_df_by_mat=np.array(tilde_sig_df.iloc[-1,:])

    return tilde_sig_df_by_mat

def auxiliar_function_all_mat(n,nbr_components,keys_n1,keys_n,new_tilde,maturity_dict):
    get_model=[auxiliar_function(j,n,nbr_components,keys_n1,keys_n,new_tilde,maturity_dict) for j in range(len(maturities))]
    get_model=np.array(get_model)
    return get_model

get_model=auxiliar_function_all_mat(n,nbr_components,keys_n1,keys_n,new_tilde,maturity_dict)
arr_dfs_bymat=np.array(Parallel(n_jobs=-1)(delayed(auxiliar_function_all_mat)(n,nbr_components,keys_n1,keys_n,new_tilde,maturity_dict) for k in tqdm(range(MC_number),desc='Getting Model at all maturities')))

prices_scaled=option_prices_splitted
Premium = prices_scaled.flatten()

index_sel_maturities=[0,1]
sel_maturities=[maturities[0],maturities[1]]
Vega_W=np.array([np.split(flat_normal_weights,len(maturities))[idx] for idx in index_sel_maturities]).flatten()
Premium1=np.array([np.split(prices_scaled,len(maturities))[idx] for idx in index_sel_maturities]).flatten()
print(Premium1)

def obj_MC_tensor(l):
    tensor_sigsde_at_mat = np.tensordot(arr_dfs_bymat, l, axes=1) + initial_price  # veloce

    # this part computes the monte carlo prices of the calls ---> output: mc_payoff_arr (quite fast)
    pay = []
    for K in strikes[0]:
        matrix_big = []
        for j in range(len(maturities)):
            payff = np.maximum(0, tensor_sigsde_at_mat[j] - K)
            matrix_big.append(payff)
        matrix = np.array(matrix_big)
        pay.append(np.mean(matrix, axis=1))
    mc_payoff_arr = np.array(pay).transpose().flatten()

    return np.sqrt(flat_normal_weights) * (mc_payoff_arr - Premium)

def get_mc_sel_mat_tensor(l):
    '''
    Input: l (np.array): a set of parameters of the model, mind the dimension
    Output: Monte Carlo prices of the model with parameters l for the selected maturities
    '''

    tensor_sigsde_at_mat=np.tensordot(arr_dfs_bymat,l,axes=1)+initial_price
    pay=[]
    for K in strikes[0]:
        matrix_big=[]
        for j in index_sel_maturities:
            payff=np.maximum(0, tensor_sigsde_at_mat[j] - K)
            matrix_big.append(payff)
        matrix=np.array(matrix_big)
        pay.append(np.mean(matrix,axis=1))
    mc_payoff_arr=np.array(pay).transpose().flatten()
    return mc_payoff_arr

#range(len(sel_maturities))

def get_iv_from_calib(calibrated_prices, strikes, maturities):
    sig_prices_mc_arr = []
    iv_calib_mc = []

    sig_prices_mc_arr = np.array(np.split(calibrated_prices, len(maturities)))
    print(sig_prices_mc_arr)
    for j in range(len(maturities)):
        for k in range(len(strikes[j])):
            iv_calib_mc.append(
                implied_vol_minimize(sig_prices_mc_arr[j, k], initial_price, strikes[j][k], maturities[j], 0,
                                     payoff="call", disp=True))

    iv_calib_arr_mc = np.array(
        [np.array(iv_calib_mc[k * len(strikes[0]):(k + 1) * len(strikes[0])]) for k in range(len(maturities))])
    return iv_calib_arr_mc, sig_prices_mc_arr

l_initial=np.random.uniform(-0.1,0.1,int(((d+1)**(n+1)-1)*D/d))

def obj_MC_tensor_selected_mat(l):
    mc_payoff_arr=get_mc_sel_mat_tensor(l)
    return np.sqrt(Vega_W)*(mc_payoff_arr-Premium1)


n=2
for t in tqdm(range(1)):
    l_initial=np.random.uniform(-0.2,0.2,int(((d+1)**(n+1)-1)*D/d))
    res1 = least_squares(obj_MC_tensor_selected_mat, l_initial,loss='linear')

calibrated_prices=get_mc_sel_mat_tensor(res1['x'])



np.save(f'ell_MC({n},{d},{rho},{MC_number},{initial_price}).npy',res1['x'])

ell_MC = res1['x']

iv_, prices_= get_iv_from_calib(calibrated_prices,strikes,sel_maturities)
np.save(f'calibrated_prices_MC({MC_number},{initial_price})',calibrated_prices)


calibrated_prices_02 = calibrated_prices
ell_calibrated_02 = ell_MC
iv_calib_arr_mc_2, sig_prices_mc_arr_2 = get_iv_from_calib(calibrated_prices_02, strikes, sel_maturities)
print(iv_calib_arr_mc_2)


for j in range(len(sel_maturities)):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Maturity T={}'.format(maturities[j]))
    ax1.plot(strikes[j], iv_calib_arr_mc_2[j], marker='o', color='c', alpha=0.7, label='SigSDE n=2')
    ax1.plot(strikes[j], iv_market[j], marker='*', alpha=0.4, color='b', label='Market')
    ax1.set_xlabel('Strikes')
    ax1.set_title('Implied volatilities')
    ax2.scatter(strikes[j], np.abs(iv_calib_arr_mc_2[j] - iv_market[j]) * 10000, color='c', alpha=0.7,
                label='SigSDE n=2')
    ax2.set_xlabel('Strikes')
    ax2.set_ylabel('Bps')
    ax2.set_title('Absolute Error in Basepoints')
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper left')
    #plt.savefig('Fit_MC_(T={}'.format(round(maturities[j], 4)) + ', all).png', dpi=100)
    plt.show()

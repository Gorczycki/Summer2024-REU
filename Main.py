import numpy
import math
import statistics
import matplotlib.pyplot as plt
import pandas
from scipy.stats import norm
import iisignature


#BS model incorporating static drift and volatility


#set t to change by .01 increments, update price, put into price vector
#perhaps set asset price as a brownian motion?
#How does volatility change as share price and time changes?
#Solve for volatility?

#Have stock price modelled as standard wiener process, and then calculate call option price from this with
#S(t) = S(0)exp{(mu - sigma^2/2)*t + sigma*W(t)}


T = 0.7 #maturity
S_0 = 49
S_t = 50 #share price
K = 55.5 #strike price
r = 0.042
stock_mu = 0.25
stock_sigma = 0.20
dt = 0.01
time_steps = int(T / dt) #number of time steps
num_brownian_motions = 5
n = 2 #degrees of signature


def W_t(num_brownian_motions, time_steps, dt):
    rng = numpy.random.default_rng(75)
    Z = rng.normal(0, 1, (num_brownian_motions, time_steps))
    W = numpy.cumsum(numpy.sqrt(dt) * Z, axis = 1)
    return W
    #returns the wiener process as vector.

#Instead, update wiener process to update as time increment changes?
    


#Sim a stock price graph with brownian motion
def stockbrownianmotion(S_0, stock_mu, W, stock_sigma):
    #from the simplification of BS, mu and sigma taken as constants, when in reality they are not.
    #dS(t) = mu*S(t)dt + sigma*S(t)dW(t)
    #S(t) = S(0)exp{(mu - (sigma^2)/2)*t + sigma*W(t)}
    #W_t2 = W_t1 + sqrt(t_2 - t_1)*Z    
    t = numpy.linspace(0, T, len(W))
    S = S_0 * numpy.exp((stock_mu - 0.5 * stock_sigma**2) * t + stock_sigma*W)
    #changed from S = S_0 * numpy.exp((stock_mu - 0.5 * stock_sigma**2) * dt + stock_sigma*W)
    #to
    #S = S_0 * numpy.exp((stock_mu - 0.5 * stock_sigma**2) * t + stock_sigma*W)

    return S


def optionpricedelta(S, K, T, r, stock_sigma):
    dplus = (1/(stock_sigma*(math.sqrt(T))))*(math.log(S/K) + (r + 0.5 * (stock_sigma**2))*(T))
    dminus = dplus - stock_sigma*(math.sqrt(T))
    firstcdf = norm.cdf(dplus)
    secondcdf = norm.cdf(dminus)
    Call = firstcdf*S - secondcdf*K*(math.exp(-r*(T)))
        

    return Call


def path_signatures(call_prices, n):
    signatures = []
    for prices in call_prices:
        prices_2d = numpy.array(prices).reshape(-1,1) #2-D data
        INDsignature = iisignature.sig(prices_2d, n)
        signatures.append(INDsignature)
    
    return signatures


W = W_t(num_brownian_motions, time_steps, dt)

stock_prices = numpy.array([stockbrownianmotion(S_0, stock_mu, W[i], stock_sigma) for i in range(num_brownian_motions)])


# Calculate call option prices over time
call_prices = []
for i in range(num_brownian_motions):
    call_prices.append([])
    for t in range(time_steps):
        T_t = T - t * dt  # Decreasing time to maturity
        if T_t <= 0:  # Avoid division by zero
            break
        S_t = stock_prices[i][t]
        call_price = optionpricedelta(S_t, K, T_t, r, stock_sigma)
        call_prices[i].append(call_price)
    
signatures = path_signatures(call_prices, n)
signature_length = iisignature.siglength(1,n)

#Define the coefficients
numpy.random.seed(123)
ell = numpy.random.randn(signature_length + 1)


#Model S_n(\ell)_t
def signature_model(signatures, ell):
    S_n = []
    for sig in signatures:
        S_n.append(ell[0] + numpy.dot(ell[1:], sig))

    return S_n

S_n_values = signature_model(signatures, ell)




def L_options(ell):
    total_loss = 0.0
    for i in range(num_brownian_motions):  # assuming N is the number of options
        # Calculate model price C_model(T_i, K_i, ell)
        model_price = call_price(T[i], K[i], ell)  # Define calculate_model_price function
        
        # Calculate squared error term
        squared_error = (gamma[i] * (market_price[i] - model_price))**2
        
        # Accumulate the loss
        total_loss += squared_error
        
    return total_loss



# Plotting
plt.figure(figsize=(12, 12))

#plt.subplot(3, 1, 1)
#for i in range(num_brownian_motions):
#    plt.plot(numpy.linspace(0, T, time_steps), W[i])
#plt.title('Wiener Process')
#plt.xlabel('Time')
#plt.ylabel('W(t)')


plt.subplot(3, 1, 1)
for i in range(num_brownian_motions):
    plt.plot(numpy.linspace(0, T, time_steps), stock_prices[i])
plt.title('Stock Price (Geometric Brownian Motion)')
plt.xlabel('Time')
plt.ylabel('S(t)')


plt.subplot(3, 1, 2)
for i in range(num_brownian_motions):
    plt.plot(numpy.linspace(0, T, len(call_prices[i])), call_prices[i])
plt.title('Call Option Price')
plt.xlabel('Time')
plt.ylabel('Call Price')


# Plot S_n(\ell)_t
plt.subplot(3, 1, 3)
plt.scatter(range(num_brownian_motions), S_n_values)  # Plotting as scatter plot
plt.title('Signature Model S_n(\ell)_t')
plt.xlabel('Index')
plt.ylabel('Value')
print(S_n_values)

plt.tight_layout()
plt.show()



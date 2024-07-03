import numpy as np
import math
import statistics
import matplotlib
import pandas
from scipy.stats import norm

#Try to configure to the standard wiener process. Draw similarities

# C(S_t,t) = N(dplus)S_t - N(dminus)Ke^(-r(T-t))

#parameters:
# K = strike price
# r = risk-free rate
# t = time
# sigma = volatility
# N = CDF of normal
# S_t = spot price of asset


S_t = 50
K = 55.5
r = 0.042
t = 0.7 #maturity
sigma = 0.70

dplus = (1/(sigma*(math.sqrt(t))))*(math.log(S_t/K) + (r + 0.5 * (sigma**2))*(t))
dminus = dplus - sigma*(math.sqrt(t))

firstcdf = norm.cdf(dplus)
secondcdf = norm.cdf(dminus)

Call = firstcdf*S_t - secondcdf*K*(math.exp(-r*(t)))

print(Call)

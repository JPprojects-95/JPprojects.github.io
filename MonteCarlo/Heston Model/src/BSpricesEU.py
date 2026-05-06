import numpy as np
from scipy.stats import norm


#############################



    #Exact prices for european call and put

def BS(S0, K, T, r, sigma):
    """
    Computes European call price using Black–Scholes formula.

    Parameters:
        S0    : initial stock price
        K     : strike price
        T     : time to maturity
        r     : risk-free rate
        sigma : volatility

    Returns:
        Call option price
    """

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put_price= call_price-S0+K*np.exp(-r*T)
    

    return call_price,put_price



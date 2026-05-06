import numpy as np

'''GBM generator'''

    #takes as inputs S0 initial price, T time interval, r interest rate riskless, sigma volatility risky,
            #n time steps, N number of paths.
    #S is an internal variable which stands for the asset price. We evolve it under the RISK-NEUTRAL dynamics.
    # Its discounted version is a martingale!
    



def ExactGBM(S0,T,r,sigma,n,N):
    dt = T / n
    S = np.zeros((n + 1, N))
    S[0] = S0

    for t in range(1, n + 1):
        deltaB=np.random.standard_normal(N)
        S[t] = S[t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * deltaB)
        
    return S # Stock price process
    

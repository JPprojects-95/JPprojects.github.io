import numpy as np
from scipy.optimize import brentq
from BSpricesEU import BS




#### Here, given a list of prices (possibly coming from Heston simulation) we compute implied volatility



def IV(S0,T,r,Strikes,DATAPRICES):           ####STRIKES IS THE SET OF STRIKES WE USE
                                                        ####DATA PRICES ARE THE CORRESPONDING PRICES

    L=len(Strikes)
    IVol=np.zeros(L)


    for i in range(0,L):
        K=Strikes[i]
        market_price = DATAPRICES[i]

        f = lambda sigma: BS(S0, K, T, r, sigma)[0] - market_price

        IVol[i] = brentq(f, 1e-6, 1.0)

    return IVol

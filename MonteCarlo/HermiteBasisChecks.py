import numpy as np
from numpy.polynomial.hermite import Hermite
from scipy.integrate import quad
import math
import matplotlib.pyplot as plt




#Hermite Polynomials and conditional expectations for Log-normal Markov processes
    




'''Hermite polynomials'''

    #They form an orthogonal basis for L^2(\mu) where $\mu$ is the Gaussian measure.
    #In our problem the process X_t is log-normally distributed
    #We can still use Hermite polynomials after coordinate transformation Y_t=(log(X_t)-mean(X_t))/std(X_t)


def H(n, x):
    coeffs = np.zeros(n + 1)  
    coeffs[n] = 1             
    H_poly = Hermite(coeffs) #Hermite(array) creates the polynomial \sum \alpha_n H_n with array=(alpha_1,dots) 
    z=x/np.sqrt(2)
    return H_poly(z)/math.sqrt((2**n)*math.factorial(n))     # Normalize and  Evaluate at x





'''Euler discretization of GBM'''

    #takes as inputs S0 initial price, T time interval, r interest rate riskless, sigma volatility risky,
            #n time steps, N number of paths.
    #S is an internal variable which stands for the asset price. We evolve it under the RISK-NEUTRAL (drift=r)
    

def EulerGBM(S0, T, r, sigma, n, N):  #In the code we also compute the evolution of the BM process along same path (needed for second method for computing delta)
    dt = T / n
    S = np.zeros((n + 1, N))
    S[0] = S0

    for t in range(1, n + 1):
        deltaB=np.random.standard_normal(N)
        S[t] = S[t-1]*(1 + r* dt + sigma * np.sqrt(dt) *deltaB)
    
        
    return S #Price 




'''Exact discretization of GBM'''


def ExactGBM(S0,T,r,sigma,n,N):
    dt = T / n
    S = np.zeros((n + 1, N))
    S[0] = S0

    for t in range(1, n + 1):
        deltaB=np.random.standard_normal(N)
        S[t] = S[t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * deltaB)
        
    return S #Price and BM arrays



'''Almost orthonormality check'''



def ALPHA(S0,T,r,sigma,n,N,Nbasis):

    S=ExactGBM(S0,T,r,sigma,n,N)
    
    logS=np.log(S[n])
    mu=np.mean(logS)
    sd=np.std(logS)

    Y=(logS-mu)/sd
  
    alpha=np.zeros((Nbasis,Nbasis))          #Matrix of inner products

    for j in range(0,Nbasis):
        for k in range(0,Nbasis):

            alpha[j,k]=np.mean(H(j,Y)*H(k,Y))     #Here we approximate the integral int_R H(j,x)*H(k,x)dmu(x)
   


    return alpha    #notice that this yields an array with the same dimension as y




#parameters

S0 = 100.0
T = 1
r = 0.05
sigma = 0.2
n = 1
N = 100000000
Nbasis=3

alphaval=ALPHA(S0,T,r,sigma,n,N,Nbasis)

print(alphaval)


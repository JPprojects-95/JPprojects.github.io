import numpy as np
from numpy.polynomial.hermite import Hermite
from scipy.integrate import quad
import math
import matplotlib.pyplot as plt




#PRICE AMERICAN CALL OPTION USING LONGSTAFF-SCHWARTZ REGRESSION+MC


    #Dimension truncation:
        #Hermite polynomials
        #Approximate Conditional Expectation operators (Via Monte Carlo)
        #Approximate the Snell envelope of the payoff process (see theory file or Pages)
        #Backward approx of optimal stopping times
    




'''Hermite polynomials, Conditional Expectation'''

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
    #S is an internal variable which stands for the asset price. We evolve it under the RISK-NEUTRAL dynamics (drift= interest rate)
    

def EulerGBM(S0, T, r, sigma, n, N):  
    dt = T / n
    S = np.zeros((n + 1, N))
    S[0] = S0

    for t in range(1, n + 1):
        deltaB=np.random.standard_normal(N)
        S[t] = S[t-1]*(1 + r* dt + sigma * np.sqrt(dt) *deltaB)

        
    return S #Price and BM arrays


'''Exact discretization of GBM'''


def ExactGBM(S0,T,r,sigma,n,N):
    dt = T / n
    S = np.zeros((n + 1, N))
    S[0] = S0

    for t in range(1, n + 1):
        deltaB=np.random.standard_normal(N)
        S[t] = S[t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * deltaB)
        
    return S #Price and BM arrays
    



'''Optimal Stopping and Snell Envelope'''


    #Payoff function

def payoff(S,K):

    
    return np.maximum(S-K,0)


    #Conditional expectation operator via MC










def CE(Z:np.ndarray, y:np.ndarray, N, Nbasis):             #Z, set of future values and y is the present value of the Markov Chain. N is the dimension of y and Z.
                                                           #Nbasis is the number of basis functions



    Gram=np.zeros((Nbasis,Nbasis))                        #Even if the empirical distribution of y i almost log-normal there are errors
                                                            #These errors make non-orthonormal effects important.
                                                            #Thus better to compute Gram matrix (See Pages)

    for i in range(0,Nbasis):
        for j in range (0,Nbasis):

            Gram[i,j]=np.mean(H(i,y)*H(j,y))            #Approx integral int_R H(i,x)*H(j,x)dmu(x)


    alpha=np.zeros(Nbasis)

    for l in range(0,Nbasis):
        alpha[l]=np.mean(Z*H(l,y))          #Here we approximate the integral int_R Z(x)*H(l,x)dmu(x)


   

    beta = np.linalg.solve(Gram, alpha)


    C=np.zeros(N)   #Initialize the conditional expectation to a one dim array


        

    for l in range (0,Nbasis):

        C=C+beta[l]*H(l,y)
   


    return C    #notice that this yields an array with the same dimension as y






    #Approximation of the Snell Envelope for payoff process

def Snell_envelope(S0, T, r, sigma, n,K, N, Nbasis):

    dt=T/n


    Z=np.zeros((n+1,N))
    
    S=EulerGBM(S0, T, r, sigma, n, N)
    
    Y=np.zeros((n+1,N))


    logS = np.log(S)                           # shape (n+1, N)
    mu = np.mean(logS, axis=1, keepdims=True)  # keeps mu as 2-dim array so we can define Y as below 
    sd = np.std(logS, axis=1, keepdims=True)   # keeps mu as 2-dim array so we can define Y as below 

    Y[1:, :] = (logS[1:, :] - mu[1:, :]) / sd[1:, :]   #The variable Y is normally distributed N(0,1) so Hermite basis
    Y[0, :] = 0.0

    
    Z=payoff(S,K)    # Initialize the envelope to consist just on the value of the payoff

    for i in range (n-1,0,-1):  #Backward from n-1 to 1

        Z[i] = np.maximum(Z[i], CE(np.exp(-r * dt)* Z[i+1], Y[i], N, Nbasis))


    Z[0]=np.maximum(Z[0],np.exp(-r*dt)*np.mean(Z[1]))


    return Z






#SIMULATION PARAMETERS: one has to be very careful between relations n,N,Nbasis

S0 = 100.0
T = 1
r = 0.05
sigma = 0.2
n = 2000
N = 1000000
Nbasis=5


K=100   #Strike price 
    


ZSnell=Snell_envelope(S0, T, r,sigma,n,K, N,Nbasis)


print(ZSnell[0,0])




































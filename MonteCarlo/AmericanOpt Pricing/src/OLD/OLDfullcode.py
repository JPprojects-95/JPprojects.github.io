import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.hermite import Hermite
import math



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
        
    return S #Stock price process
    







'''Hermite polynomials, Conditional Expectation'''

    #They form an orthogonal basisfor L^2(\mu) where $\mu$ is the Gaussian measure.
    #In our problem the discounted stock price process S_t is log-normally distributed
    #We can use Hermite polynomials as basis after coordinate transformation Y_t=(log(X_t)-mean(X_t))/std(X_t).


def H(n, x):
    coeffs = np.zeros(n + 1)  
    coeffs[n] = 1             
    H_poly = Hermite(coeffs) #Hermite(array) creates the polynomial \sum \alpha_n H_n with array=(alpha_1,dots) 
    z=x/np.sqrt(2)
    return H_poly(z)/math.sqrt((2**n)*math.factorial(n))     # Normalize and  Evaluate at x










''''' COMPUTE CONDITIONAL EXPECTATIONS '''''''''''



def CE(Z:np.ndarray, y:np.ndarray, N, Nbasis):             #Z, set of future values and y is the present value of the Markov Chain. N is the dimension of y and Z.
                                                           #Nbasis is the number of basis functions



    Gram=np.zeros((Nbasis,Nbasis))                        #Even if the EMPIRICAL DISTRIBUTION of y i almost log-normal there are errors
                                                            #These errors make non-orthonormal effects important.
                                                            #Thus better to compute Gram matrix (See Pages)

    for i in range(0,Nbasis):
        for j in range (0,Nbasis):

            Gram[i,j]=np.mean(H(i,y)*H(j,y))            #Approx integral int_R H(i,x)*H(j,x)dmu(x)


    alpha=np.zeros(Nbasis)

    for l in range(0,Nbasis):
        alpha[l]=np.mean(Z*H(l,y))          #Here we approximate the expectation E(Z_{i+1} H(l,S_i))


   

    beta = np.linalg.solve(Gram, alpha)


    C=np.zeros(N)   #Initialize the conditional expectation to a one dim array


        

    for l in range (0,Nbasis):

        C=C+beta[l]*H(l,y)
   


    return C    #notice that this yields an array with the same dimension as y







''''''''' RECOMPUTE THE PROCESS FROM CONDITIONAL EXPECTATIONS '''''''''''''''''''''


    
def Reconstruction(S0, T, r, sigma, n, N, Nbasis):

  
    Z=np.zeros((n+1,N))
    
    S=ExactGBM(S0,T,r,sigma,n,N)
    
    Y=np.zeros((n+1,N))


    logS = np.log(S)                           # shape (n+1, N)
    mu = np.mean(logS, axis=1, keepdims=True)  # keeps mu as 2-dim array so we can define Y as below 
    sd = np.std(logS, axis=1, keepdims=True)   # keeps sd as 2-dim array so we can define Y as below 

    Y[1:, :] = (logS[1:, :] - mu[1:, :]) / sd[1:, :]   #The variable Y is normally distributed N(0,1) so Hermite basis is fine
    Y[0, :] = 0.0

    
    Z=S  # Initialize 

    for i in range (n-1,0,-1):  #Backward from n-1 to 1

        Z[i] =  CE(Z[i+1], Y[i], N, Nbasis)


    return S,Z






''''''''''' SNELL ENVELOPE '''''''''''


 #Payoff function

def payoff(S,K):

    
    return np.maximum(K-S,0)






###### ENVELOPE

#Approximation of the Snell Envelope for payoff process

def Snell_envelope(S0, T, r, sigma, n,K, N, Nbasis):

    dt=T/n


  
    
    S=ExactGBM(S0,T,r,sigma,n,N)
    
    Y=np.zeros((n+1,N))


    logS = np.log(S)                           # shape (n+1, N)
    mu = np.mean(logS, axis=1, keepdims=True)  # keeps mu as 2-dim array so we can define Y as below 
    sd = np.std(logS, axis=1, keepdims=True)   # keeps sd as 2-dim array so we can define Y as below 

    Y[1:, :] = (logS[1:, :] - mu[1:, :]) / sd[1:, :]   #The variable Y is normally distributed N(0,1) so Hermite basis
    Y[0, :] = 0.0

    
   


    V=np.exp(-r*T)*payoff(S,K)    # Initialize the VALUE PROCESS to consist just on the (discounted) value of the payoff

    for i in range (n-1,0,-1):  #Backward from n-1 to 1

        V[i] = CE(V[i+1], Y[i], N, Nbasis)            #This would be the discounted value process for European claim, just for comparison


    V[0]=np.mean(V[1])

    


    Z=np.zeros((n+1,N))
    P=np.zeros((n+1,N))

    Z=np.exp(-r*T)*payoff(S,K)    # Initialize the envelope to consist just on the  (discounted) value of the payoff
  
     
    for i in range (n-1,0,-1):  #Backward from n-1 to 1

        P[i] = np.exp(-r*dt*i)*payoff(S[i],K)

        Z[i] = np.maximum(P[i], CE(Z[i+1], Y[i], N, Nbasis))     #Ensure that, always, the envelope is above the current (discounted) payoff.


    Z[0]=np.maximum(Z[0],np.mean(Z[1]))



    return V,Z










'''''''''''''''''''''' SIMULATION '''''''''''''''''''''


S0=100
r=0.05
sigma=0.2
T=1
n=200
N=50000
Nbasis=4


K=100.1

##### NORMALIZE THE RW ######


S=Reconstruction(S0, T, r, sigma, n, N, Nbasis)[0]



###### RECONSTRUSCTED RW######

R=Reconstruction(S0, T, r, sigma, n, N, Nbasis)[1]



####### VALUE PROCESS

V=Snell_envelope(S0, T, r, sigma, n,K, N, Nbasis)[0]
Z=Snell_envelope(S0, T, r, sigma, n,K, N, Nbasis)[1]


''''''''' VISUALIZATION '''''''''''''''




############ PATH RECONSTRUCTION

TIME1=n//10




TIME1=n//10


counts1, bins1 = np.histogram(S[TIME1], bins=20, density=True)
counts2, bins2 = np.histogram(R[TIME1], bins=20, density=True)

# Bin centers
centers1 = 0.5 * (bins1[1:] + bins1[:-1])
centers2 = 0.5 * (bins2[1:] + bins2[:-1])

# Plot as lines
plt.figure()
plt.plot(centers1, counts1, label=f"Simulation")
plt.plot(centers2, counts2, label=f"Reconstruction")

plt.xlabel("Stock Price")
plt.ylabel("Density")
plt.title("Reconstruction from CE")
plt.legend()
plt.show()


############## VALUE PROCESSES



TIME1=n//10


counts1, bins1 = np.histogram(V[TIME1], bins=20, density=True)
counts2, bins2 = np.histogram(Z[TIME1], bins=20, density=True)

# Bin centers
centers1 = 0.5 * (bins1[1:] + bins1[:-1])
centers2 = 0.5 * (bins2[1:] + bins2[:-1])

# Plot as lines
plt.figure()
plt.plot(centers1, counts1, label=f"VALUE PROCESS")
plt.plot(centers2, counts2, label=f"ENVELOPE")

plt.xlabel("Stock Price")
plt.ylabel("Density")
plt.title("VALUE AND SNELL ENVELOPE")
plt.legend()
plt.show()





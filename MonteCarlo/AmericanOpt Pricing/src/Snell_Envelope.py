import numpy as np
from GBM import ExactGBM
from Conditional_Expectation import CE


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

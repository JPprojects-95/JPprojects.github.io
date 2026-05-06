import numpy as np



####### Here we simulate paths of the Heston model #####

        ### This is a stochastic volatility model in which volatility follows a square root diffusion process.
        ### The parameters for this process are a,b,sigma.
        ### The stock price then follows a GBM with stochastic volatility.
        ### An extra parameter is the correlation rho between the Wiener processes driving this system of SDEs

        # S0, Vol0 stock initial price and volatitility initial value
        # T,n length of time interval and number of subdivisions,
        # r riskless interest rate
        # a, b,sigma a,b are mean reverting parameters and sigma=volatility driving the square root process
        # MUST SATISFY Feller's condition!
        # rho correlation between Wiener 



def Heston(S0,Vol0,T,r,sigma,a,b,rho,n,N):
    dt = T / n
    S = np.zeros((n + 1, N))
    Vol=np.zeros((n+1,N))

    
    S[0] = S0
    Vol[0]= Vol0

    for t in range(1, n + 1):
        deltaB1=np.random.standard_normal(N)
        deltaB2=np.random.standard_normal(N)

        S[t] = S[t-1]*(1+r*dt+np.sqrt(np.maximum(Vol[t-1],0))*np.sqrt(dt)* deltaB1)
        Vol[t]=Vol[t-1]+a*(b-Vol[t-1])*dt + sigma * np.sqrt(np.maximum(Vol[t-1],0))*np.sqrt(dt)*(rho* deltaB1 + np.sqrt(1-rho**2)*deltaB2)
        
        
    return S,Vol # Stock price process and vol process



    

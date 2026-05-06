import numpy as np
import math
from numpy.polynomial.hermite import Hermite


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



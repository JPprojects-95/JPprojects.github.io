import numpy as np
from Hermite import H




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




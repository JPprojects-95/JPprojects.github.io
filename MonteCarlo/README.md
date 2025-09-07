Some implementations of the Monte Carlo method for option pricing:


  -American options via the Longstaff-Schwarz algorithm:
  
      -Pricing of Call options (same price as European, no benefit from early exercise due to convexity)
      -Auxiliary file for checking almost orthogonality of Hermite basis when empirical distribution.

  -European option via discretization of the GBM SDE:
  
      -Computation of option price
      -Computation of delta via two different formulas: direct definition and Malliavin weight. For T \to 0 we observe variance reduction: the direct definition presents smaller variance (see Theory file about variance reduction).
  
  -Zero-coupon bonds via: 
  
      - discretization of the sqrt diffusion SDE,
      
      -using the exact transition distribution kernel

# Store some functions used in the code
import math
import numpy as np
import scipy.special as sp

# Plasma dispersion function
def Z(ze): 
  z = complex(0,1.0)*np.sqrt(np.pi)*sp.wofz(ze)
  return z

def zet(ze): 
  z = complex(0,1.0)*np.sqrt(np.pi)*sp.wofz(ze)
  return z
# Calculate the paralle integral Z_n(zeta)
def p(ze,n):
  w  = 1.0 + ze*Z(ze)
  if   n==0:
    z = Z(ze)
  elif n==1:
    z = w
  elif n==2:
    z = ze*w
  elif n==3:
    z = (1.+2.*ze**2*w)/2.0
  elif n==4:
    z = ze*(1.+2.*ze**2*w)/2.0
  else:
    sys.exit("Error Message:Wrong n for zeta function!")
  return z

# This section Calculate the derivative of parallel integral
def dp(ze,n):
  w  = 1.0 + ze*Z(ze)
  dz = -2.0*w
  dw = Z(ze)+ze*dz
  if   n==0:
    dp  = dz
  elif n==1:
    dp  = dw
  elif n==2:
    dp  = w + ze*dw
  elif n==3:
    dp  = 2.*ze*w+ze**2*dw
  elif n==4:
    dp  = (1.+2.*ze**2*w)/2.0+\
          ze*(2*ze*w+ze**2*dw)
  else:
    sys.exit("Error: Wrong n for zeta function derivative!")
  return dp


# Derivative of exponentially scaled modified 
# Bessel function 
# sp.ive(n,a_s)= I_n(a_s)*exp(-a_s)
# i means ith derivative of the function
def dive(n,a,i):
  a = float(a)
  # No perpendicular component
  if a ==0:
    if   i == 1:
      fz = 0.5*(sp.ive(n-1,a)+sp.ive(n+1,a))-sp.ive(n,a)
    elif i == 2:
      fz = 0.25*sp.ive(n-2,a)-sp.ive(n-1,a)+1.5*sp.ive(n,a)\
           -sp.ive(n+1,a)+0.25*sp.ive(n+2,a)
    else:
      sys.exit("Error:i should be 1 or 2 for dive function!")
   
  # with perpendicular component
  else:
    if   i == 1:
      fz = (n/a-1)*sp.ive(n,a)+sp.ive(n+1,a)
    elif i == 2:
      fz = ((n**2-n)/a**2-2*n/a+1.)*sp.ive(n,a)+\
           ((2*n+1.0)/a-2)*sp.ive(n+1,a)+sp.ive(n+2,a)
    else:
      sys.exit("Error:i should be 1 or 2 for dive function!")
  return fz

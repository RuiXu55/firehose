import zetaf as f
import numpy as np
import scipy.special as sp
from scipy.linalg import det
from numpy import linalg as LA

w = np.ones(3,dtype =complex)    # eigenvalue w[i]
v = np.ones((3,3),dtype=complex) # eigenvectors v[:,i]
def DSP(z0,*data):
    global w,v

    beta,krho0,mu,va_c,Del,theta = data
    tau = beta[0]/beta[1]   # Ti_prl/Te_prl
    # first ion, second electron
    zeta = [complex(z0[0],z0[1]),complex(z0[0],z0[1])*np.sqrt(mu*tau)]
    krho = [krho0, -krho0*np.sqrt(mu*Del[1]/tau/Del[0])]
    
    D    = np.zeros((3,3),dtype=complex)
    m    = 2
    coef =[1.,mu]
    # sum over species
    for i in range(0,2):
        D1    = np.zeros((3,3),dtype=complex)
        al = (krho[i]*np.sin(theta))**2/2.0    # alpha_s = (k_prp*w_prp/Omega_s)^2/2.
        Omega = np.sqrt(Del[i])/krho[i]/np.cos(theta) # Omega_s/(k_prl*w_prl)
        # sum over n
        for n in range(-m,m+1):
            ze = zeta[i] - n*Omega
            An = (Del[i]-1.)/zeta[i]+(ze*Del[i]+n*Omega)/zeta[i]*f.Z(ze)
            Bn = 1. + ze*An
            
            j = complex(0,1.0)
            D1[0,0] += n**2*sp.ive(n,al)*An/al
            D1[0,1] += j*n*f.dive(n,al,1)*An
            D1[0,2] += n*sp.ive(n,al)*Bn*2./(np.sqrt(Del[i])*krho[i]*np.sin(theta))
            D1[1,1] += (n**2/al*sp.ive(n,al)-2*al*f.dive(n,al,1))*An
            D1[1,2] += -j*f.dive(n,al,1)*Bn*krho[i]*np.sin(theta)/np.sqrt(Del[i])
            D1[2,2] += 2.*sp.ive(n,al)*Bn*ze/Del[i]
        
        D1[1,0]  = -D1[0,1]
        D1[2,0]  =  D1[0,2]
        D1[2,1]  = -D1[1,2]
        
        D += D1*Omega**2/zeta[i]*coef[i]
    
    # add (va_c term comes from displacement current)
    D += va_c**2*np.identity(3)
    
    tmp = 1./(zeta[0]**2*beta[0])
    D[0,0] += -tmp
    D[0,2] += np.tan(theta)*tmp
    D[1,1] += -(1.+np.tan(theta)**2)*tmp
    D[2,0] += np.tan(theta)*tmp
    D[2,2] += -np.tan(theta)**2*tmp
    
    res = det(D)
    w, v = LA.eig(D)
    return (res.real,res.imag)

import numpy as np
import numpy.ma as ma
import dispersion as dsp
from scipy.optimize import root
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import matplotlib as mpl
font = {'family' : 'serif',
    'weight' : 'normal',
        'size'   : 18}
mpl.rc('font', **font)

#-----------------------------------------------------------
#-----------------------------------------------------------
num   = 20
krho  = np.linspace(.01,40.,num)   # k*w_prp/Omega_s
fzeta = np.ones((num),dtype=complex) # omega/(k_prl*w_prl)
ztmp = np.ones((num),dtype=complex) # omega/(k_prl*w_prl)
maxgrow = np.ones((num))
# parameters
Del   = [1.0,0.9]       # T_prp/T_prl (ion+e)
mu    = 1./100.        # me/mi
beta  = [2.,20.]      # beta_i & beta_e in parallel direction
va_c  = 1.e-3         # V_A/c
#va_c  = np.sqrt(0.03433)
theta = 60
theta*= np.pi/180.0



ztmp  = np.ones(num,dtype=complex) # omega/(k_prl*w_prl)
for i in range(0,num):
  zeta = complex(0.,0.01)
  data  = (beta,krho[i],mu,va_c,Del,theta)
  try:
      sol = root(dsp.DSP,(zeta.real,zeta.imag),args=data,method='hybr',tol=1e-5)
      fzeta[i] = complex(sol.x[0],sol.x[1])
      #x, y =  fsolve(FHIDSP.DSP, (zeta.real,zeta.imag),args=data,xtol=1.e-1)
      #fzeta[i] = complex(x,y)
  except ValueError:
      try:
        zeta = complex(0.,0.008)
        sol = root(dsp.DSP,(zeta.real,zeta.imag),args=data,method='hybr',tol=1e-5)
        fzeta[i] = complex(sol.x[0],sol.x[1])
      except ValueError: 
        zeta = fzeta[i-1]+complex(-0.0001,-0.0001)

ztmp = fzeta*(krho*np.cos(theta)/np.sqrt(Del[0]))*mu
zmasked=ma.masked_where(np.abs(ztmp.imag)>=1.,ztmp)

lam = 2.*np.pi*np.sqrt(Del[0]*beta[0])/krho/np.sqrt(mu)
kva = krho/np.sqrt(Del[0]*beta[0])*np.sqrt(mu)
#plt.plot(lam,zmasked.imag)
#plt.plot(kva,zmasked.imag*np.sqrt(mu/va_c))
plt.plot(kva,zmasked.imag)
plt.xlabel(r'$kc/\omega_{pe}$')
#plt.xlabel(r'$\lambda(c/\omega_e)$')
plt.ylabel(r'$\gamma/\Omega_{e}$')
#plt.title(r'$\lambda_{max}$')
plt.axis([0,0.4,0,0.04])

fig = plt.gcf()
#plt.savefig('ana_dsp_beta20_del09_theta60.pdf',format='pdf',dpi=300,bbox_inches='tight')
plt.show()

#!/usr/bin/env python
# coding: utf-8

# # Problem 3 - Answers
# 
# **Angular Eigenstate of the SHO**
# 
# In your lecture notes you have seen the mathematical derivation of an angular momentum eigenstate of the 2D simple harmonic oscillator, in the case where the potential is spherically symmetric. This eigenstate a superposition of the first two excited energy eigenstates such that
# 
# $$ \psi = u_{10} + i u_{01}. $$
# 
# Here $u_{n_x, n_y}(x, y) = u_{n_x}(x) \times u_{n_y}$ where $u_{n_x}$ is an energy eigenfunction of the 1D SHO.
# 
# 1. Assume an electron is confined in a simple harmonic potential well with $\omega_0 = 2\times10^{16}$ s$^{-1}$. Produce an animation showing the time evolution of the energy eigenstates $u_{10}(x, y, t)$ and $i u_{01}(x, y, t)$. What effect does the factor of $i$ have on the eigenstate $i u_{01}$?
# 
# The following code can be used to create an animated contour plot:
# ```python
# #First set up the figure
# fig, ax = plt.subplots(1, 1)
# ax1.set_xlim((x[0],x[-1]))
# ax1.set_ylim((y[0], y[-1]))
# quad1 = ax1.pcolormesh(x, y, z, shading='gouraud',
#                        vmin=<min z>, vmax=<max z>)  # this line is vital to ensure the colour map stays consistent
# 
# def init():
#     quad1.set_array([])
#     return quad1
# 
# def animate(i):
#     z = <z at time t>
#     quad1.set_array(z.ravel())
#     return quad1
#     
# anim = animation.FuncAnimation(fig, animate)
# plt.close(fig)  # Include this line to prevent two figures appearing in the notebook
# 
# HTML(anim.to_jshtml())  # This line allows you to see the animation in the notebook
# ```
# 
# 

# In[1]:


# Import packages and set constants
import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib import cm
from IPython.display import HTML
from matplotlib import animation
from scipy.special import eval_hermite as hermite

hbar = 1.0545718e-34
h = 6.62607004e-34
m_e = 9.1094e-31
omega = 2e16


# In[2]:


# Use functions previously seen in class to define enegy eigenstates for SHO
def un(x, n, omega=1.0, m = m_e):
    """
    Energy eigenvalues
    x - 1D position in meters
    n - integer quantum number
    """
    
    norm = np.power(m * omega / (np.pi * hbar), 0.25)
    norm /= np.sqrt(2**n * math.factorial(n))
    alpha = np.sqrt(m * omega / hbar)
    y = np.real(alpha * x)  # This line has been altered so that the function un can take complex arguments
    
    un = norm * hermite(n, y) * np.exp(-0.5 * np.power(y, 2.0))
    
    return un


def En(n, omega=1.0, m = m_e):
    """
    Energy eigenvalues
    n - integer quantum number
    """
    
    En = hbar * omega * (n + 0.5)
    return En


def lengthScale(omega=1.0, m = m_e):
    """
    Characteristic length scale for the SHO
    """
    alpha = np.sqrt(m * omega / hbar)
    return 1.0 / alpha


def psin(x, t, n, omega=1.0, m =m_e):
    """
    individual eigenstates
    x - 1D position in meters
    n - integer quantum number
    """
    
    psi = un(x, n, omega, m) * np.exp(-1j * En(n, omega, m) * t / hbar)
         
    return psi


# In[3]:


omega_0 = 2e16
scale = lengthScale(omega, m=m_e)
x = np.linspace(-5*scale, 5*scale, 100, dtype='complex128')
y = np.linspace(-5*scale, 5*scale, 100, dtype='complex128')
xx, yy = np.meshgrid(x, y)
t = np.linspace(0, 4*np.pi/omega, 200, dtype='complex128')
nframes = len(t)

# Create functions for eigenstates
def u10(t):
    wavefunction = psin(xx, t, 1, omega=omega_0, m=m_e) * psin(yy, t, 0, omega=omega_0, m=m_e)
    return wavefunction

def u01(t):
    wavefunction = psin(xx, t, 0, omega=omega_0, m=m_e) * psin(yy, t, 1, omega=omega_0, m=m_e)
    return wavefunction

# First set up the figure, the axis, and the plot elements we want to animate
plt.rcParams['image.cmap'] = 'RdBu_r'
fig, [ax1, ax2] = plt.subplots(1, 2, figsize = [12, 6])
ax1.set_xlim((x[0],x[-1]))
ax1.set_ylim((y[0], y[-1])) 
ax2.set_xlim((x[0],x[-1]))
ax2.set_ylim((y[0], y[-1]))
quad1 = ax1.pcolormesh(xx, yy, np.real(u10(0)), shading='gouraud',
                       vmin=-np.max(np.real(u10(0))),
                       vmax=np.max(np.real(u10(0))))
quad2 = ax2.pcolormesh(xx, yy, np.real(np.complex(0, 1)*u01(0)), shading='gouraud',
                       vmin=-np.max(np.real(u01(0))),
                       vmax=np.max(np.real(u01(0))))

def init():
    quad1.set_array([])
    quad2.set_array([])
    return quad1, quad2

def animate(i):
    time = t[i]
    u10_plot = u10(time)
    u01_plot = np.complex(0, 1) * u01(time)
    quad1.set_array(np.real(u10_plot.ravel()))
    quad2.set_array(np.real(u01_plot.ravel()))
    return quad1, quad2
    
anim = animation.FuncAnimation(fig,animate,frames=nframes,interval=50,blit=False,repeat=False)
plt.close(fig)  # Include this line to prevent two figures appearing in the notebook

HTML(anim.to_jshtml())  # This line allows you to see the animation in the notebook


# The factor of $i$ results in a phase shift of $\pi / 2$, so $u_{10}$ is at its maximum when $i u_{01}$ is zero.
# 
# 2. Plot an animation showing the real part of the full time dependent wavefunction.

# In[4]:


omega_0 = 2e16
scale = lengthScale(omega, m=m_e)
x = np.linspace(-5*scale, 5*scale, 100, dtype='complex128')
y = np.linspace(-5*scale, 5*scale, 100, dtype='complex128')
xx, yy = np.meshgrid(x, y)
t = np.linspace(0, 4*np.pi/omega, 200, dtype='complex128')
nframes = len(t)

# Create functions for eigenstates
def u10(t):
    wavefunction = psin(xx, t, 1, omega=omega_0, m=m_e) * psin(yy, t, 0, omega=omega_0, m=m_e)
    return wavefunction

def u01(t):
    wavefunction = psin(xx, t, 0, omega=omega_0, m=m_e) * psin(yy, t, 1, omega=omega_0, m=m_e)
    return wavefunction

# Create function for angular momentum eigenstate and probability density
def ang_eigenstate(t):
    return u10(t) + np.complex(0, 1)* u01(t)


# First set up the figure, the axis, and the plot element we want to animate
plt.rcParams['image.cmap'] = 'RdBu_r'
fig, ax1 = plt.subplots(1, 1, figsize = [6, 6])
ax1.set_xlim((x[0],x[-1]))
ax1.set_ylim((y[0], y[-1]))
quad1 = ax1.pcolormesh(xx, yy, np.real(ang_eigenstate(0)), shading='gouraud',
                       vmin=-np.max(np.real(ang_eigenstate(0))),
                       vmax=np.max(np.real(ang_eigenstate(0))))

def init():
    quad1.set_array([])
    return quad1

def animate(i):
    time = t[i]
    ang_to_plot = ang_eigenstate(time)
    quad1.set_array(np.real(ang_to_plot.ravel()))
    return quad1
    
anim = animation.FuncAnimation(fig,animate,frames=nframes,interval=50,blit=False,repeat=False)
plt.close(fig)  # Include this line to prevent two figures appearing in the notebook

HTML(anim.to_jshtml())  # This line allows you to see the animation in the notebook


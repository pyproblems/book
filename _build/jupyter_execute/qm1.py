#!/usr/bin/env python
# coding: utf-8

# # Problem 1 - Answers
# 
# **Time evolution before and after measurement**
# 
# An arbitrary wavefunction is confined in an infinite square well of width L. Some time later a measurement is made of the particle's position. Assume that the uncertainty on this position measurement is Gaussian in form with a standard deviation of $L/10$.
# 
# 
# 1.Produce an animation showing the time evolution of this wavefunction before and after the position measurement.
# 
# Before the measurement the time development of an arbitrary wavefunction $\psi(x, t)$ is given by
# 
# $$ \psi(x, t) = \sum_n a_n u_n(x) e^{-iE_n t / \hbar},  $$
# 
# where $u_n(x)$ are the energy eigenstates and $a_n$ can be calculated using the overlap integral
# 
# $$ a_n = \int u_n^*(x) \psi(x) dx.$$
# 
# When the position measurement is made, the wavefunction collapses to an eigenstate of position. Which in this case is a Gaussian function with standard deviation $L/10$. The position is randomly selected from a probability distribution of the form $|\psi(x, t)|^2$. 
# 
# After this point the wavefunction evolves as before but with an updated set of $a_n$.

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from scipy.interpolate import interp1d
from scipy.stats import norm
from IPython.display import HTML
from matplotlib import animation

hbar = 1.05e-34
m_e = 9.11e-31

# Use infinite square well functions from previous examples in class
def un(x, n, L=1.0):
    """
    Energy eigenvalues
    x - 1D position in meters
    n - integer quantum number
    """
    
    norm = np.sqrt(2/L)
    kx = n * np.pi * x / L
    un = norm * np.sin(kx)
    
    un = np.where(np.logical_and(x>0, x<L), un, 0)
    return un

def En(n, L=1.0, m =m_e):
    """
    Energy eigenvalues
    n - integer quantum number
    """
    
    En = np.pi**2 * n**2 * hbar**2 / (2.0 * m * L**2)
    return En

def psin(x, t, n, L=1.0, m =m_e):
    """
    individual eigenstates
    x - 1D position in meters
    n - integer quantum number
    """
    
    psi = un(x, n, L) * np.exp(-1j * En(n, L, m) * t / hbar)
         
    return psi

def measurement(PDF, size=None):
    """
    Given a PDF return the indx corresponding to a random measurement
    """
    #renormalise to ensure sum to one
    PDF = PDF / np.sum(PDF)
    
    indx = np.random.choice(np.arange(len(PDF)), p=PDF, size=size)
    
    return indx

def overlapIntegral(n, x, psi):
    """
    Calculate overlap integral
    n - integer quantum number
    x - x coordinates of arbitrary wavefunction
    psi - y coordinates of arbitrary wavefunction
    """
    f_psi = interp1d(x, psi)  # Create function from arbitrary wavefunction
    def integrand(dummy):
        return np.conj(un(dummy,n)) * f_psi(dummy)  # Define the integrand for overlap integral
    an = scipy.integrate.quad(integrand, x[0], x[-1])  # Integrate within limits of wavefunction
    return an[0]


# In[2]:


x = np.linspace(-0.5, 1.5, 10000)
# Define arbitrary wavefunction
y = np.sqrt(0.5)*un(x, 1) + np.sqrt(0.25)*un(x, 2) + np.sqrt(0.25)*un(x, 4) 
t = np.linspace(0, 4000, 500)
expansions = 20  # Number of expansions to use in overlap integral

nframes = len(t)
measurement_index = np.int(nframes/2)  # Time to make position measurement
measurement_time = t[measurement_index]
accuracy = 0.1  # Uncertainty on position measurement

# First set up the figure, the axis, and the plot element we want to animate
fig, [ax1, ax2] = plt.subplots(1, 2, figsize = [12, 6])
plt.close(fig)  # Include this line to prevent two figures appearing in the notebook
ax1.set_xlim((x[0],x[-1]))
ax2.set_xlim((x[0],x[-1]))
ax1.set_ylim((-5, 5))
ax2.set_ylim((-0.1, 20))
line1, = ax1.plot([], [], lw=2)
line2, = ax2.plot([], [], lw=2)
ax1.set_xlabel("x")
ax1.set_ylabel("$psi(x)$")
ax2.set_xlabel("x")
ax2.set_ylabel("$|psi(x)|^2$")
text = ax1.text(-0.3, 4.5, '')
fig.tight_layout()

lines = []
lines.append(line1)
lines.append(line2)

# Calculate the co-efficients for the initial wavefunction using an overlap integral
ans = []
ans_new = []
for i in range(1, expansions):
    an = overlapIntegral(i, x, y)
    ans.append(an)

# initialization function: plot the background of each frame
def init():
    for line in lines:
        line.set_data([], [])
    return lines

# animation function.  This is called sequentially
def animate(i):
    time = t[i]
    #calculate wavefunction
    if i < measurement_index:  # Time evolution before measurement
        y_t = np.zeros(np.shape(x), dtype='complex128')
        for k in range(1, expansions):
            y_t += ans[k-1]*psin(x, time, k)     # Calculate wavefunction at time t 
        prob_t = np.real(np.conjugate(y_t)*y_t)  # by propegating eigenstates forwards in time
    
    # At time of measurement wavefunction is measured to be at a position randomly selected
    # from the PDF. This is assumed to be gaussian with a specified uncertainty.
    elif i == measurement_index:   # Wavefunction at time of measurement
        y_t = np.zeros(np.shape(x), dtype='complex128')
        for k in range(1, expansions):
            y_t += np.complex(ans[k-1], 0)*psin(x, time, k)
        x0 = x[measurement(np.real(np.conjugate(y_t)*y_t), size=None)]  # randomly selected x position
        y_t = np.where(np.logical_and(x>0, x<1), norm.pdf(x, x0, accuracy), 0)  # wavefunction is gaussian indside well
        prob_t = np.real(np.conjugate(y_t)*y_t)                                 # and zero elsewhere
        print("Particle measured at x = ", x0)
        # Recalculate co-efficients for new wave function
        for i in range(1, expansions):
            an = overlapIntegral(i, x, y_t)
            ans_new.append(an)
        ax1.plot(x, y_t, "--")  # plot the possition measurement permenantly on the figure
        ax2.plot(x, prob_t, "--")
        
    else:  # Time evolution after measurement with new set of coefficients
        time = t[i] - measurement_time
        y_t = np.zeros(np.shape(x), dtype='complex128')
        for k in range(1, expansions):
            y_t += np.real(ans_new[k-1]*psin(x, time, k))
        prob_t = np.real(np.conjugate(y_t)*y_t)
        
    #update the lines and text
    lines[0].set_data(x, y_t)
    lines[1]. set_data(x, prob_t)
    text.set_text("t = %0.0f" % np.round(time, -1))
    return lines

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=nframes, interval=40, blit=True)

HTML(anim.to_jshtml())  # This line allows you to see the animation in the notebook


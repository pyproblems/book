# Problem 3

**Angular Eigenstate of the SHO**

In your lecture notes you have seen the mathematical derivation of an angular momentum eigenstate of the 2D simple harmonic oscillator, in the case where the potential is spherically symmetric. This eigenstate a superposition of the first two excited energy eigenstates such that

$$ \psi = u_{10} + i u_{01}. $$

Here $u_{n_x, n_y}(x, y) = u_{n_x}(x) \times u_{n_y}$ where $u_{n_x}$ is an energy eigenfunction of the 1D SHO.

1. Assume an electron is confined in a simple harmonic potential well with $\omega_0 = 2\times10^{16}$ s$^{-1}$. Produce an animation showing the time evolution of the energy eigenstates $u_{10}(x, y, t)$ and $i u_{01}(x, y, t)$. What effect does the factor of $i$ have on the eigenstate $i u_{01}$?
2. Plot an animation showing the real part of the full time dependent wavefunction.

The following code can be used to create an animated contour plot:
```python
#First set up the figure
fig, ax = plt.subplots(1, 1)
ax1.set_xlim((x[0],x[-1]))
ax1.set_ylim((y[0], y[-1]))
quad1 = ax1.pcolormesh(x, y, z, shading='gouraud',
                       vmin=<min z>, vmax=<max z>)  # this line is vital to ensure the colour map stays consistent

def init():
    quad1.set_array([])
    return quad1

def animate(i):
    z = <z at time t>
    quad1.set_array(z.ravel())
    return quad1
    
anim = animation.FuncAnimation(fig, animate)
plt.close(fig)  # Include this line to prevent two figures appearing in the notebook

HTML(anim.to_jshtml())  # This line allows you to see the animation in the notebook
```

# Problem 1

**Simulating binary collisions**

This problem will have you simulate binary collisions between two particles. 

## Rutherford Scattering

*Rutherford scattering* is the elastic scattering of charges interacting via a Coulomb potential. Here we will be modelling Rutherford's original experiment - scattering of fast-moving alpha particles off of stationary Gold nuclei.

a) Define some relevant constants for this simulation, including the masses and charges of Helium and Gold nuclei, and the initial speed of the alpha particles (use a speed of $10^7$ms$^{-1}$). Also define a function describing the Coulomb interaction force:

$$ \mathbf{F}_{C} = \frac{Qq}{4\pi\epsilon_0r^3}\mathbf{r} $$

b) There are two important simulation parameters which wish to estimate: the size of the simulation domain, and the time step of the simulation. The simulation domain should be much larger than the length scale of the interaction force, and the time step should be small enough that a particle takes many time steps to traverse this length scale. 

Approximate the length scale of the Coulomb interaction as the radius at which the kinetic energy of the incident alpha particle and the Coulomb potential energy are equal, and use this to set the simulation domain size and time step.

c) Consider a simulation domain in which a Gold nucleus is located at the origin, and has your chosen extent in both directions along the z-axis. The domain will have no fixed extent in the x and y axes. 

Define a `collide` function which simulates the trajectory of an alpha particle in this simulation domain. The alpha particle will have a fixed initial velocity pointing in the z-direction. It should have an initial position on one simulation boundary. The 'impact parameter' or distance between the particle's initial position and the z-axis should be an argument of your function. The trajectory of the alpha subject to the Coulomb force should be numerically integrated until the particle leaves the simulation domain (you may use a simple Euler method to do this.) The function should return the trajectory of the particle.

d) Plot these trajectories for a range of different impact parameters. 

## Hard Sphere Scattering

An alternative simple model of binary collision is that of scattering off of a hard sphere. 

a) Modify your `collide` function to accept an additional argument R, the radius of a hard-sphere atom model. Without removing your implementation of the Coulomb force, add a condition which checks if the alpha hits the surface of the sphere, and if so is reflected. 

```{admonition} Hint
:class: dropdown, tip
The equation for reflecting an alpha particle's velocity $\mathbf{v}$ off of a surface with unit normal vector $\mathbf{n}$ is:
$$ \mathbf{v}_r = \mathbf{v}_i - 2(\mathbf{v}_i\dot\mathbf{n})\mathbf{n}  $$
```

b) To simulate scattering off of an uncharged hard sphere, set the charge on the Gold nucleus to zero and give it a radius of $10^{-13}$m. Plot the trajectories of alpha particles for a range of impact parameters.

c) Now we will consider the interaction of particles with a charged hard sphere. Return the charge on the particle at the origin to the Gold nuclear charge. Now simulate the trajectories of alphas with a range of impact parameters. Try changing the radius of the hard sphere to see how this changes how much the situation resembles hard sphere scattering vs Rutherford scattering.
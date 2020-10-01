# Problem 2

In this problem, you will take the code developed for *method 3* in the previous problem, and modify it to simulate interacting particles in a box. We will not be implementing functionality for dealing with particle-particle or particle-wall collisions - instead we will work with repelling particles that are confined within a box by repulsive fields. The force describing interactions between particles is:

$$ \mathbf{F}_{ip} = \frac{A}{r^3} \exp{\left(\frac{-r}{K}\right)} \mathbf{r} $$

Each particle also experiences a force due to each of the four walls of the box:

$$ \mathbf{F}_{W} = \frac{B}{r_{\perp}^3} \exp{\left(\frac{-r_{\perp}}{L}\right)} \mathbf{r}_{\perp} $$

where $r_{\perp}$ is the perpendicular distance between the particle and the wall. 

This pairing of exponentially decaying repulsive potentials between particles and other particles, and between particles and the boundaries, is used in modelling a range of physical situations from a crowd moving through a corridor to charged dust particles in a plasma. 

```{note}
Start with following parameters for the simulation: $A=1$, $B=10$, $K=0.1$, $L=0.1$, and a timestep $\delta t=0.001$. Once you have a working simulation, you can modify these and see the effect.
```

## Setup

a) Modify your initialise function so that the initial positions are not totally random, but are instead guaranteed to all be initially a certain distance apart (this is to prevent numerical issues caused by particles initially being on top of each other). You could do this by sampling initial positions from a grid. You should also change the initial velocities from zeros to being sampled from a normal distribution. The standard deviation of this distribution should be an argument to the function.

b) Adapt your method 3 function to implement the above particle-particle and particle-wall forces. Rename this function `step`.

c) Make an animation of how a system of 100 particles evolves over 100 time steps. Check that this behaves as you would expect.

## Measuring Pressure

The pressure that is exerted on each wall of the box due to interactions with the particles may be found by summing over the particle-wall forces.

a) Modify your `step` function to calculate the total force exerted on each wall. As the walls have unit length, this is identical to the pressure on the wall. `step` should return the sum of the pressures on each wall (as well as the updated positions and velocities).

b) Run your simulation of 100 particles for 2000 time steps. Plot how the pressure changes over this time. 

c) You should find that after some time the system comes to an equilibrium and the pressure oscillates about some mean value. Define a function `pressure` that takes two arguments: the number of particles, and the standard deviation of the random distribution of particle speeds. It should return the mean pressure.

d) Use your  `pressure` function make a plot of how pressure varies with the number of particles in the box. Does it vary as you expect? Extract parameter(s) to characterise the dependence. 

e) Now make a plot of how pressure varies with the typical speed of particles in the box. Does it vary as you expect? Extract parameter(s) to characterise the dependence. 
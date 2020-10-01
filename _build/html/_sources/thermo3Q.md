# Problem 3

In the problem, we will be once again building on the simulation developed in the previous problems. We will be taking advantage of our approach of simulating particles at discrete time steps to add in other kinds of forces on particles. 

a) One type of force commonly considered in simulating macroparticles is drag. Modify your step function to add an additional 'Stokes drag' force on each particle:

$$ F_D = -\gamma \mathbf{v}  $$

You can start with a value of $\gamma=10$. Animate your simulation to see how adding drag impacts your simulation. 

b) You'll have noticed that a drag force causes the particles to slowly lose energy until they come to rest. This is not physical - the <a href="https://en.wikipedia.org/wiki/Fluctuation-dissipation_theorem">fluctuation-dissipation theorem</a> says that the physical process causing drag, or 'dissipation' (in this case, collision with microparticles) will also cause a 'fluctuation'. In the case of a macroparticle in a fluid, this is Brownian motion. 

Add Brownian motion to your simulation. This can be represented as a force term which, at each time step, is randomly sampled from a 2D Gaussian distribution. Look up the <a href="https://en.wikipedia.org/wiki/Langevin_equation">Langevin equation</a> for more details on this. You can start by using a value of 100 as the standard deviation of the Gaussian.

Make an animation and observe the effects.

c) Changing the standard deviation of the Gaussian Brownian noise, you are effectively modifying the 'temperature' of the particle system. Repeat your animation with different 'temperatures' and note the effect.

d) Clearly, the particle system is more 'disordered' at higher temperatures. A good way of visualising the order of a particle system is to plot its *correlation function*. Make and compare plots of the correlation function for different temperatures.
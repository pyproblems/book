# Problem 3

**Particle scattering and collection in an attractive potential**

The 'mixed' scattering considered in the previous problems, involving a charged hard sphere, has not been seen by the author in describing particle scattering. However, it does appear in a different context - studying the collection and scattering of ions and electrons by a charged spherical 'dust' particle in a plasma. In this problem we will study the interaction between positively charged ions and a negatively charged dust particle in more detail. 

You may use your `collide` function from the previous problems. But the physical parameters will change somewhat. A dust particle may have a charge of -10,000e and a radius of 10$\mu m$. The incident particle might be a Hydrogen ion. The initial velocity of this particle should be added as an argument to the `collide` function, but you may use 10,000ms$^{-1}$ as a typical value to determine the simulation domain size and time step.

a) Modify your `collide` function to return True/False to indicate whether the incident ion collides with the dust grain surface, or not. 

b) Define a function `bmax` which takes the radius of the dust grain and the initial speed of the ion as arguments. It should return the maximum impact parameter for which ions are 'collected' (collide with the dust surface). 

```{admonition} Hint
:class: dropdown, tip
An efficient way to implement this would be to use a form of <a href="https://en.wikipedia.org/wiki/Bisection_method">bisection method</a>, to repeatedly bisect an interval of impact parameter space until you get within a certain precision of where the return of `collide` goes from True to False.
```

c) Pick a range of values for initial ion speeds (e.g. speeds between 10$^3$ms$^{-1}$ and 10$^5$ms$^{-1}$). Evaluate the value of $b_{max}$ for each of these speeds, and make a plot. How does this compare against the function:

$$ b = R\sqrt{1-\frac{2Qq}{4\pi\epsilon_0Rmv^2}}  $$

This expression, which can be derived straightforwardly from conservation of energy and momentum, is key to the 
<a href="https://doi.org/10.1017/S0022377800008345">Orbital Motion Limited (OML)</a> model of dust charging in a plasma, and while never being stricted valid is a useful model.

d) In reality, the potential around a charged body in a plasma does not look like a Coulomb potential, but instead takes the form of the <a href="https://en.wikipedia.org/wiki/Debye_length">
Debye–Hückel potential</a>, which is an exponentially decaying Coulomb potential with a decay length $\lambda_D$:

$$ \mathbf{F}_{D} = \frac{Qq}{4\pi\epsilon_0r^3} \exp{\left(\frac{-r}{\lambda_D}\right)} \mathbf{r} $$

Define a function that specifies the Debye–Hückel potential, and use it to solve for and plot the $b_{max}$ vs $v$ profiles for values of $\lambda_D$ that are smaller than, larger than, and equal to the radius of the dust particle. How do these profiles compare to the OML model?
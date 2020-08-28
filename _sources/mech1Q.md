# Question 1 - Problems

This question concerns the properties of the potential function found in the file <a href="_data/Potential.txt">Potential.txt</a>.

a) Plot the potential given by the data

b) Find the equilibrium point(s) of the potential and show if they are stable or unstable

```{note}
You can use <samp>scipy.interpolate.interp1d</samp> to create a function which you can then solve using <samp>fsolve</samp>. Use the <samp>xtol</samp> parameter to specify a sensible value for the tolerance of the solution found by <samp>fsolve</samp>.
```

c) Find and plot the maximum energy of particle bound in this potential

d) Find and plot the allowed region for this bound particle
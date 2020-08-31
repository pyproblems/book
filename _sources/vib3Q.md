# Problem 3

Consider a guitar string of length 61 cm, density 3.09 kgm$^{-1}$ and tension 56.4 N that is struck by a pick 53.4 cm from the top of the guitar, imparting an instantaneous velocity of 50 ms$^{-1}$ to that area of the string. Use the fourth order Runge-Kutta method to simulate the evolution in time of the wave along this string, and display it using the FuncAnimation function from matplotlib's animation module. 

```{admonition} Hint
:class: dropdown, tip
To turn a PDE like the wave equation into an ODE-like system of equations that can then be solved by Runge-Kutta you can split the string up into discrete points, and then use a finite difference method to approximate the laplacian at each point along the string. When setting your time and space divisions, be sure to ensure $v\frac{\delta t}{\delta x} \leq 1$, as if this is not the case then the solution will not be numerically stable.
```
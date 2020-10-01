# Problem 1

There are generally many routes to simulating a physics problem in Python. Some of these will be more efficient, particularly those that rely on functions in libraries like numpy and scipy that are built on well optimised, compiled code. This problem builds functionality that will be used in Problems 2 and 3 for doing thermodynamics simulation. But we will try doing the same thing in three different ways, and will see that we can get 10-100 times speed-up by doing it the 'right' way. 

## The Problem

Many thermodynamic principles can be tested numerically by simulating a large number of interacting particles. If you have done the Thermodynamics problem in 2nd Year Computing, you will have seen how simulating the collisions of hard spheres can reproduce certain important phenomena. However, the approach to modelling used there does not allow implementation of interparticle forces, which are important in real fluids. 

In the following sections, your task will be to implement a simulation of $n$ identical particles, interacting via a simple Coulomb-like force:

$$ \mathbf{F} = \frac{\alpha}{r^3}\mathbf{r} $$

A simple simulation suitable for testing will use $\alpha=1$, will sample initial particle positions randomly in the unit square, and will give particles zero initial speed. A simulation time step of 0.001 will be suitable. 

Write a function `initialise` which takes the number of particles $n$ as an argument. This function should return an $n\times 2$ numpy array of random 2D initial particle positions, and an $n\times 2$ array of zeros, for the initial particle velocities. 

In each of the following sections, define a function that takes the number of particles $n$, and $n\times 2$ position and velocity arrays as arguments. These functions used the method specified to step the position and velocity arrays one step forward in time. 

## Nested for loops

Use a nested pair of for loops, each iterating over all $n$ particles. For a given pair of different particles i and j, calculate the force of j acting on i. Sum over all the forces acting on particle i, and use this total force to do an Euler step and update the positions and velocities. 

## SciPy pdist function

Look up the documentation for the <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html">scipy.spatial.distance.pdist</a> function. This is a useful function which calculates the matrix of distances between each pair of points in an N-dimensional space. 

```{note}
You will see that the return of pdist is a 'condensed' matrix. To convert to a conventional square matrix, use scipy.spatial.distance.squareform.
```

pdist will allow us to implement this functionality without using any for loops. However, we have a problem to work around: pdist returns pairwise scalar distances, but to find the forces we require pairwise *displacements*. pdist will take as an argument a function defining a custom metric, however this must still be a scalar. Try to figure out you can use multiple calls to pdist to work around this issue. 

```{admonition} Hint
:class: dropdown, tip
You can access the upper and lower triangles of a square numpy array using the np.triu and np.tril functions.
```

## Array flattening

A third way of approaching this problem is to use only a single for loop rather than two nested loops. This can be done using the functionality of numpy arrays.

This will require taking the difference between appropriate slices of the position array. To assign these displacements into a square matrix, use of the numpy 'flat' property of an array will be required. You can deal with each dimension separately.

```{admonition} Hint
:class: dropdown, tip
Your for loop will be over the n-1 diagonal slices of one half of the square matrix of displacements. 
```

## Testing

a) Generate initial position and velocity arrays for 5 particles. Test these on the three different methods you've implemented, and check that you get the same results for the iterated velocity for all 3 methods.

b) Create a list of values for $n$, ranging from around 10 to around 1000. Write a loop over these values, and for each use the time.time() function to measure the time each of the three methods takes to run one time step using this number of particles. Make a plot comparing these three methods. 
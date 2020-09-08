# Problems

In these exercises, we will develop a basic simulation to explore how charged particles interact with magnetic fields.
Some of these exercises involve producing visualisations with the external plotting library `matplotlib`.
As such, it is recommended that you import its `pyplot` interface with
```{code-block} python
import matplotlib.pyplot as plt
```
In addition, you may want to use the `numpy` library to provide arrays which can be used as vectors. 
`numpy` is typically imported with
```{code-block} python
import numpy as np
```

Lastly, as we will be simulating small charged particles moving through space, it might be helpful to define the following constants to use throughout the exercises:

```{code-block} python
echarge = 1.60217662e-19  # electron charge
emass = 9.10938356e-31    # electron mass
vperm = 1.25663706212e-6  # vacuum permittivity
```

## 1. Defining some vector products

### 1.a. Dot product
**EXERCISE**: Define a function with signature
```{code-block} python
dot(v1, v2)
```
that takes in two 3-component vectors (i.e 1-dimensional `numpy` arrays of length 3) `v1` and `v2` and returns their vector dot product.

### 1.b. Testing the dot product
**EXERCISE**: Test your `dot` function by working out some dot products of some simple 3-component vectors by hand and comparing these with the function output.

### 1.c. Cross product
**EXERCISE**: Define a function with signature
```{code-block} python
cross(v1, v2)
```
that takes in two 3-component vectors (i.e 1-dimensional `numpy` arrays of length 3) `v1` and `v2` and returns their vector cross product, also as a 3-component vector).

### 1.d. Testing the cross product
**EXERCISE**: Test your `cross` function by working out some cross products of some simple 3-component vectors by hand and comparing these with the function output.

## 2. Visualising some magnetic fields
As we will be working with magnetic fields, we will need a convenient way to represent them in our code. A magnetic field is a vector field; that is, it assigns a vector $\mathbf{B(\mathbf{x}, t)}$ to each point in space $\mathbf{x}$ at each time $t$. As such, a convenient way to represent our magnetic field will be a function of the form `Bfield(pos, t)` that takes in a position vector `pos` and a time `t` and returns the appropriate 3-component magnetic field vector (again, as a  1-dimensional `numpy` arrays of length 3)

### 2.a Defining some field functions
**EXERCISE**: Define
- a function `Bfield_constant(pos, t)` that returns a magnetic field of magnitude $1\mathrm{T}$ along the positive $z$-direction, for any position `pos` and time `t`
- a function `Bfield_spacevarying(pos, t)` that returns a magnetic field, oriented along the positive $z$-direction, with magntitude $1\mathrm{T}/||\mathbf{x}||$ (where $||\mathbf{x}||$ is the magntitude of the position vector `pos`) for any time `t` (it is up to you if and how you handle the singularity at $||\mathbf{x}||=0$)
- a function `Bfield_timevarying(pos, t)` that returns a magnetic field, parrallel to the $z$-axis, with $z$-component $1\mathrm{T}\cos(2\pi t)$ for any position vector `pos`.

All of these field functions we've defined take both arguments `pos` and `t`, even if the calculations don't actually use them. This *consistent interface* will be useful later on when we want to write code that works with arbitrary magnetic fields.

### 2.b. Defining a function to generate a grid of points parallel to the $xy$-plane
We are nearly ready to visualise the magnetic fields that we've defined. While we could in principle do this in three dimensions, it might be clearer to visualize 2-dimensional slices of space instead. Furthermore, while we could in principle pick arbitrary slices, it will be much easier to choose planes that are aligned with two of our co-ordinate axes. This is what we will do here.

We want to define a surface, parrallel to the $yz$-plane that extend from $y=y_0$ to $y=y_\mathrm{max}$ and from 
$z=z_0$ to $z=z_\mathrm{max}$, and which has a particular constant $x$ co-ordinate. To treat this computationally, we will need to discretize this surface into small rectangles of sides $\delta y$ and $\delta z$. We can then represent the surface by a collection of the points that are the centres of these rectangles. For example, a surface defined by 
$x=0.123$
$y_0=-0.2, y_\mathrm{max}=0.2$
$z_0=0.0, z_\mathrm{max}=0.4$
and which is discretized with $\delta y=\delta z = 0.2$ would be represented by the points $(0.123, -0.1, 0.1)$, $(0.123, 0.1, 0.1)$, $(0.123, -0.1, 0.3)$, $(0.123, 0.1, 0.3)$.

**EXERCISE**: Define a function `gen_grid(x, y0, z0, ymax, ymax, zmax, dyz)` that returns a python `list` of points (i.e position vectors as `numpy` arrays) that make up a planar surface parrallel to the $yz$-plane, as described above. `x` should be the constant $x$-co-ordinate, and `dyz` should give the discretization value $\delta y=\delta z$ (using a single value for simplicity means that the small 'rectangles' will really be squares.)

**EXERCISE**: Using `gen_grid`, generate some surfaces and plot the points using `plt.scatter` to check that they look as you expect. 
If you don't know how to turn your `list` of $(x, y, z)$ points into something more easily plottable by `plt.scatter`, the following will be helpful:
```{code-block} python
grid = gen_grid(...) # arguments replaced by ... for brevity 
x, y, z = zip(*tuple(grid))
plt.scatter(y, z, ...)
```
If you don't understand what this code is doing, look online for information about the `zip` function, tuple-packing and tuple-unpacking. This is a very useful pattern.

### 2.c. Visualising the magnetic fields
**EXERCISE**: Using `gen_grid`, generate a a surface defined by 
$x=0.0$
$y_0=-2.0, y_\mathrm{max}=2.0$
$z_0=-2.0, z_\mathrm{max}=2.0$
and which is discretized with $\delta y=\delta z = 0.2$. For each point in the grid, call `Bfield_constant` (at time 0) and store the resulting field vector (which should be the same for each point). Plot these field vectors as their respective points using `plt.quiver`. The call to `plt.quiver` will look something like this: 
```{code-block} python
plt.quiver(y, z, By, Bz, ...)
```
where `By` and `Bz` are sequences of the $y$ and $z$ components of the field vectors respectively. You may need to use the `zip` trick again to prepare `By` and `Bz`.

**EXERCISE**: Repeat the above exercise for `Bfield_spacevarying` at time $t=0.0$, and for `Bfield_timevarying` at times $t=0.0$, $t=0.25$, $t=0.5$. Do the visualizations look as you expect? If the time-varying case at $t=0.25$ is producing an unexpected results, look at the documentation for `plt.quiver}`; in particular, look at the `scale` parameter.

## 3. Writing and testing an integrator for vector differential equations
The motion of a particle with charge $q$ under the influence of a magnetic field $\mathbf{B}$ is described by the equation

```{math}
:label: ac_lorentzforce
\mathbf{F} = q\mathbf{v}\times\mathbf{B}
```
where $\mathbf{v}=\frac{\mathrm{d}}{\mathrm{d}t}\mathbf{x}$ is the velocity of the particle and $\mathbf{F}$ is the resulting force the particle experiences. 
Since we also have that $\mathbf{F}=m\frac{\mathrm{d}^2}{\mathrm{d}t^2}\mathbf{x}$ where $m$ is the mass of the particle, we ultimately have to solve the differential equation
```{math}
:label: ac_2ndode
\frac{\mathrm{d}^2}{\mathrm{d}t^2}\mathbf{x} = \frac{q}{m}(\frac{\mathrm{d}}{\mathrm{d}t}\mathbf{x})\times\mathbf{B}(\mathbf{x}, t)
```
As such, we will need to write an integrator. 

Eq. {eq}`ac_2ndode` is a 2nd-order vector differential equation, but we can simplify things a bit by turning it into a 1st-order one. 
To do this, we can use the facts that $\frac{\mathrm{d}}{\mathrm{d}t}\mathbf{x} = \mathbf{v}$ and that $\frac{\mathrm{d}}{\mathrm{d}t}\mathbf{v} = \frac{\mathrm{d}^2}{\mathrm{d}t^2}\mathbf{x}=q\mathbf{v}\times\mathbf{B}(\mathbf{x}, t)$ to rewrite Eq. {eq}`ac_2ndode` as
```{math}
:label: ac_concatvec
\frac{\mathrm{d}}{\mathrm{d}t}
\Big(
\begin{array}{ll}
\mathbf{x} \\ \mathbf{v}
\end{array}
\Big)
= \Big(
\begin{array}{c}
\mathbf{v} \\ \mathbf{\frac{q}{m}\mathbf{v}\times\mathbf{B}(\mathbf{x}, t)}
\end{array}
\Big).
```
where 
$\Big(
\begin{array}{ll}
\mathbf{x} \\ \mathbf{v}
\end{array}
\Big)$
is the 6-component vector formed by concatenating $\mathbf{x}$ and $\mathbf{v}$. 

So, we just need to write a 1st-order vector integrator. A simple method to use is the explicity Euler method, which solves a differential equation
```{math}
:label: gen_de
\frac{\mathrm{d}}{\mathrm{d}t}\mathbf{u}(t) = f(\mathbf{u}(t), t).
```
for the vector $\mathbf{u}(t)$, with boundary condition $\mathbf{u}(t_\mathrm{init})=\mathbf{u}_\mathrm{init}$, by iterating the step
```{math}
:label: disc_de
\mathbf{u}(t_{n+1}) = \mathbf{u}(t_n) +  (\delta t)f(\mathbf{u}(t_n), t_n),
```
where $\delta t$ is some chosen step size.

### 3.a. Writing the integrator
**EXERCISE** Write a function of the form
```{code-block} python
euler(vec_init, t_init, t_final, dt, diff_func, ...)
```
which takes in some initial vector `vec_init`, an initial time `t_init`, a final time `t_final`, a step-size `dt` and a function `diff_func` which plays the role of $f$ in Eqs. {eq}`gen_de` and {eq}`disc_de` (that is, it is the right-hand-size of the differential equation). The function `euler` should use the Euler method to integrate the differential equation defined by `diff_func` and return both a `list` of times $t_n$ and a `list` of vectors $\mathbf{u}(t_n)$ corresponding to the solution at those times. Avoid making reference to specific vector lengths; `euler` should work with arbitrarily-sized vectors.

### 3.b. Testing with a 1st-order differential equation
To check that our Euler integrator is working, it is useful to check against a simple 1st-order differential equation, and it doesn't get much simpler than the exponential function.

**EXERCISE** Use the following setup 
```{code-block} python
def diff_func(vec, t):  # simple first order case
	return 2.0*vec

t_init = 0.0
t_final = 2.0
dt = 0.01
vec_init = np.array([1.0])  # boundary condition - the function 
							# takes value 1.0 at t\_init
```
to test your integrator by integrating the equation $\frac{\mathrm{d}}{\mathrm{dt}}g(t)=2g(t)$ with the boundary condition $g(0)=1$. Note that, although this is really a scalar differential equation, we are defining `def_init` here as a 1-component vector, so that `euler` will accept it.

**EXERCISE** Try plotting the result using `plt.plot` (look at the documentation online if you don't know how to use it). Does it looks how you expect? Try plotting the exact solution on the same axes to compare. How does varying `dt` affect it?

### 3.c. Testing with a 2nd-order differential equation
**EXERCISE** Now try with a 2nd-order version of the differential equation, where `vec` will now refer to the vector containing the value of $g(t)$ and of its first derivative. That is, redefine `vec\_init = np.array([1.0, 2.0])` so that it contains both boundary conditions, $g(0)=1.0$ and $\frac{\mathrm{d}}{\mathrm{dt}}g(t)\Big|_{t=0}=2$, and redefine `diff_func` to act appropriately on this larger vector.

## 4. Simulating a charged test particle moving in a magnetic field
We are now almost ready to perform a simulation. All that is left to do is create the appropriate differential equation to solve with our integrator.

### 4.a. Writing the differential equation
**EXERCISE** Consider a particle of mass $m_e$ (electron mass) and charge $e$ (electron charge), initially at the origin and moving with a speed $v=10^{11} \mathrm{m}\mathrm{s}^{-1}$ in the positive direction along the $x$-axis under the influence of the constant magnetic field we defined in `Bfield_constant`. 

Write an appropriate `diff_func` function that corresponds to the right-hand-side of Eq. {eq}`ac_concatvec`. Remember, it will need to be a function that operates on a 6-component vector which contains the position and velocity co-ordinates together.

### 4.b. Performing a simulation

**EXERCISE** Using the ingredients we've built so far, simulate the particle moving under the influence of the constant magnetic field for a duration $t_\mathrm{final}=2\pi\frac{m_e}{(1\mathrm{T})e}$ (what's so special about this number?). Be sure to choose a time step $\delta t$ that is small compared to this. Plot the resulting trajectory in the $xy$-plane using `plt.plot` (why is the motion confined to thr $xy$-plane?). Does it looks how you expect?

### 4.c. Another simulation

**EXERCISE** Repeat the previous exercise, but this time with a particle whose charge is $-e$. Does the result differ in the way you expect?

## 5. Visualising the magnetic field induced by a moving charge
When a charged particle moves through space, it induces a magnetic field in the space around it. This phenomenon is governed by the equation
```{math}
:label: ac_inducedfield
\mathbf{B}_\mathrm{ind}(\mathbf{x}) = \frac{q\mu_0}{4\pi} \mathbf{v}_\mathrm{particle}\times\Bigg(\frac{\mathbf{x}-\mathbf{x}_\mathrm{particle}}{|\mathbf{x}-\mathbf{x}_\mathrm{particle}|^2}\Bigg)
```
where $\mathbf{x}$ is a point in space, $q$ is the particle charge, $\mu_0$ is the vacuum permeability, and $\mathbf{x}_\mathrm{particle}$ and $\mathbf{v}_\mathrm{particle}$ are the positions and velocities of the particle respectively.

### 5.a. Writing the induced field function
Consider a particle of mass $m_e$ (electron mass) and charge $e$ (electron charge), positioned at the co-ordinates $(0.1\mathrm{m}, 0.1\mathrm{m}, 0.1\mathrm{m})$ and moving with a speed $v=10^{26} \mathrm{m}\mathrm{s}^{-1}$ in the positive direction along the $x$-axis.

**EXERCISE** Write a function `Bfield_particle(pos, t)` that corresponds to Eq. {eq}`ac_inducedfield` as applied to the particle described above.

### 5.b. Visualising the induced field
**EXERCISE** Repeat the exercise in part 2.c. using the magnetic field defined by `Bfield_particle`. Does the visualisation look how you expect?

## 6. Gauss' law for magnetism

Gauss' law for magnetism, one of the four Maxwell's equations, states that (in integral form) that for any closed surface $S$ in space,
```{math}
:label: ac_gausslaw
\intop_S \mathbf{B} \cdot \mathrm{d}\mathbf{A} = 0
```
where $\mathbf{B}$ is the magnetic field evaluated at each point on the surface and $\mathrm{d}\mathbf{A}$ is the vector surface element (i.e a vector whose magnitude is an infinitesimal area and whose direction is normal to the surface and pointing out).
This integral is the magnetic flux through the surface $S$.
In this part, we will explore this law, starting by creating a cubic surface centred on the origin.
You will want to use finer grids than you used for the field visualisations, because we will be approximating a surface integral.

### 6.a. Creating the cubic surface
**EXERCISE** Figure out a way to generalise your your `gen_grid` function from part 2.b. to create a planar surface parrallel to a chosen co-ordinate plane (e.g parrallel to the $xz$-plane).

**EXERCISE** Using your generalised `gen_grid` function, generate the six faces of a cube of side $1.0\mathrm{m}$, centred on the origin.

### 6.b. Creating the area element vectors
**EXERCISE** For each of the surfaces created above, create an appropriate area element vector $\delta\mathbf{A}$.
These will be vectors with magnitude equal to the area of the discrete squares that make up the surface, and with direction that is facing out of the cube (i.e away from the origin).

### 6.c. Calculating the fluxes
Consider the particle whose induced field we visualised in part 5.b. We want to verify that Gauss' law for magnetism holds for this induced field.

**EXERCISE** For each of the six faces, compute the discrete form of the integral in Eq. {eq}`ac_gausslaw`. That is, compute

```{math}
 \sum_{i} \mathbf{B}(\mathbf{x}_i) \cdot \delta\mathbf{A} 
```
for each surface, where $\mathbf{x}_i$ are the points defining the surface.
Which surface fluxes are non-zero? Does this make sense? Is the total flux through the closed surface zero? What do you think are the limits to this approach?
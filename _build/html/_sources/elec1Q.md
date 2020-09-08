# Problem 1

Your task is to simulate the motion of 3 electrically charged particles. To solve this problem you will need to use finite difference methods. Specifically, you will use a forward Euler solver which you will code yourself, as well as a pre-packaged solver from SciPy.

## Background

Below, you will be given background information required to solve the problem. The basic physics will be explained first, followed by a description of the finite difference and computational methods you will need to employ.

### Physics

#### Coulomb's Law

The electrostatic force between two point charges is given by the following relation:

$$ |\mathbf{F}| = k_e \frac{q_1 q_2}{r^2}, $$

where $|\mathbf{F}|$ is the magnitude of the force acting on the charges, $k_e$ is Coulomb's constant $\mathrm{(}k_e = 8.990 \times 10^9\ \mathrm{N\, m^2\, C^{-2}}\mathrm{)}$, $q_x$ is the charge of point charge $x$, and $r$ is their separation. If $q_1$ and $q_2$ have the same sign, the force is repulsive along the vector linking the charges, and vice versa.

#### Electric Field

To calculate the force on a given point charge, Coulomb's law has to be applied to all other charges in the system. This leads to the definition of the electric field:

$$ \mathbf{E}_1 = \frac{\mathbf{F}_1}{q_1}, $$

where $\mathbf{F}_1$ is the force acting on the point charge of interest, and $q_1$ is its charge.

The force can be calculated as

$$ \mathbf{F}_1 = \sum_{i \neq 1} k_e \frac{q_1 q_i}{|\mathbf{r}_{i1}|^2} \mathbf{\hat{r}}_{i1} $$

from the definition of the force above written in vector form, with $r_{i1}$ being the vector from $r_i$ to $r_1$.

#### Particle Acceleration

We will assume that each of our point charges has a constant mass $m_i$. Then

$$ \frac{\mathbf{F}_i}{m_i} = \bf{a}_i = \frac{\mathrm{d} \mathbf{v}_i}{\mathrm{d}\:\! t} = \frac{\mathrm{d^2} \mathbf{x}_i}{\mathrm{d}\:\! t^2}, $$

where $\mathbf{F}_i$ is the force acting on point charge $i$, $\mathbf{a_i}$ its acceleration, $\mathbf{v_i}$ its velocity, and $\mathbf{x_i}$ its position.

We are trying to simulate the **future** motion of the particles given their **current** state. Thus, we want to determine $\mathbf{x}_i^{t+h}$, $\mathbf{v}_i^{t+h}$, and $\mathbf{a}_i^{t+h}$ from $\mathbf{x}_i^{t}$, $\mathbf{v}_i^{t}$, and $\mathbf{a}_i^{t}$, where $\mathbf{x}_i^{t+h}$ refers to the position of point charge $i$ at time $t+h$, where $h$ is the integration time step.

This second order initial value problem can be rewritten as a coupled system of first order equations:

$$ \frac{\mathrm{d} \mathbf{x}^t}{\mathrm{d}\:\! t} = \mathbf{v}^t $$

$$ \frac{\mathrm{d} \mathbf{v}^t}{\mathrm{d}\:\! t} = \mathbf{a}^t $$

where the initial positions $\mathbf{x}^{t_0}$ and velocities $\mathbf{v}^{t_0}$ are the known initial conditions.

### Computational Methods

#### Finite Differences

Using a Taylor series,

$$ \mathbf{x}^{t+h} = \mathbf{x}^{t} + \frac{\mathrm{d} \mathbf{x}^t}{\mathrm{d}\:\! t} h + O(h^2), $$

where the discretisation error is seen to scale with $h^2$. Since

$$ \frac{\mathrm{d} \mathbf{x}^t}{\mathrm{d}\:\! t} = \mathbf{v}^t, $$

$$ \mathbf{x}^{t+h} \approx \mathbf{x}^t + h \mathbf{v}^t, $$

with $\mathbf{v}^t$ approximated by

$$ \mathbf{v}^t = \frac{\mathbf{x}^{t+h} - \mathbf{x}^t}{h}. $$

Similarly,

$$ \mathbf{a}^t = \frac{\mathbf{v}^{t+h} - \mathbf{v}^t}{h} $$

and

$$ \mathbf{v}^{t+h} \approx \mathbf{v}^t + h \mathbf{a}^t. $$

An implementation of the above scheme is one of the simplest methods of numerically integrating differential equations, known as the explicit forward Euler method.

#### SciPy Solver

Using the SciPy <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html">solve_ivp</a> function will yield significantly better results than using the first-order explicit Euler scheme above, because it is capable of using higher order integration methods with a much lower discretisation error. It also dynamically adjusts the integration step size to further reduce the error.

In order to use this function, you must rewrite your force calculation function to adhere to a function signature akin to that below:

```{code-block} python
def int_fun(t, y, masses, charges):
    """Differential equations for the system of point charges.

    Args:
        t (float): Time.
        y (numpy.ndarray): Array containing the state variables,
            (x_1_x, x_1_y, v_1_x, v_1_y, ...), the position and velocity
            components of all the charges.
        masses (numpy.ndarray): (N,) array of masses in kg.
        charges (numpy.ndarray): (N,) array of charges in C.

    Returns:
        iterable: Rates of change of the state variables
        (v_1_x, v_1_y, a_1_x, a_1_y, ...).

    """
    # Your code here ...
```

Note especially that the function **must** take a single float (the time) as its first parameter, even if the function never uses this parameter.

The function can then be used like this (see <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html">solve_ivp</a> for a thorough description of the function parameters):

```{code-block} python
sol = solve_ivp(
    partial(int_fun, masses=masses, charges=charges),
    [0, 1e-4],
    init_state,
    t_eval=np.linspace(0, 1e-4, 2000),
    rtol=1e-7,
)
# The solutions at different times are then contained in sol.y
```

You might have to set the <samp>rtol</samp> parameter to a low number like <samp>1e-7</samp> (as above) to achieve realistic results (this is especially relevant to the question in the next notebook)!

## Your Task

Your task is to simulate the trajectories of the three particles described below. You will need to use both the forward Euler solver, as well as the <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html">solve_ivp</a> function from SciPy to do this.

You are expected to use the animation code provided further below (amongst other visualisations if you wish) to evaluate your two solutions.

You will also have to compare the solutions arrived at using the two different solvers. One way to contrast these solutions could be to plot the x-position of one particle over time.

Your final task is to identify a general property of the system that remains constant throughout and show (graphically or otherwise) that this is the case.

### Initial Conditions

The three point charges you will simulate will start at rest at $(x, y)$ positions (m) $(2, 1)$, $(1, 3)$, and $(3, 2)$ with charges (C) $1$, $-4$, and $2$, and masses (kg) $1$, $3$, and $2$ respectively (all units are SI).

The temporal range of integration should encompass at least $0$ to $10^{-4}$ seconds.

### Coding Hints

#### Initialisation

One way to initialise the system:

```{code-block} python
import numpy as np
positions = np.array([[2, 1, 3], [1, 3, 2], dtype=np.float64)
velocities = np.zeros_like(positions)
masses = np.array([1, 3, 2], dtype=np.float64)
charges = np.array([1, -4, 2], dtype=np.float64)
```

#### Writing a function to calculate the electric field

Since the force acting on a given point charge is given by

$$ \mathbf{F}_j = q_j \mathbf{E}_j, $$

with $\mathbf{E}_j$ given by

$$ \mathbf{E}_j = \sum_{i \neq j} k_e \frac{q_j}{|\mathbf{r}_{ij}|^2} \mathbf{\hat{r}}_{ij} $$

it makes sense to define a function that calculates $\mathbf{E}$ as defined above.

While the exact implementation is up to you, this could look something like this:

```{code-block} python
def E(x, y, q, r):
    """Electric field.

    Args:
        x (float): X position(s).
        y (float): Y position(s).
        q (float): Charge(s).
        r (iterable of float): (x, y) position(s) of the point charge(s).
            If an array is given, it should be a (2, N) array where N
            is the number of point charges.

    Returns:
       float: Electric field vectors at every point in `x` and `y`. The
       shape of this array is the same shape as `x` and `y` with an
       added initial dimension.

    """
    # Your code here ...
```
While you could write a function to calculate the electric field from a single point charge, it makes sense to vectorise this function so that it accepts multiple charges. Both methods should yield the same result, since the individual electric fields from multiple charges may be added together to yield the combined field.

#### Animation Code

The code below is meant to be run inside a Jupyter notebook.

Should you choose run it elsewhere, please see <a href="https://matplotlib.org/api/animation_api.html">animation_api</a> for further information.

Using <a href="https://pypi.org/project/tqdm/2.2.3/">tqdm</a> here is completely optional. It provides a straightforward way to judge the time remaining for the animation from within the notebook. Should you wish to run the code without it, simply delete the line using tqdm and unindent everything from <samp>def animate(i)</samp> up to and including <samp>js_output</samp> by one level.

```{code-block} python
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from matplotlib import animation
from tqdm import tqdm

# Just to illustrate what shape this has to be.
# It could just as well be a list of (2, 3) arrays!
all_positions = np.random.random((10, 2, 3))

charges = np.array([1, -4, 2], dtype=np.float64)

fig, ax = plt.subplots()
scat = ax.scatter(*all_positions[0], c=charges)

all_xs = [x for step_positions in all_positions for x in step_positions[0]]
all_ys = [y for step_positions in all_positions for y in step_positions[1]]

ax.set_xlim([np.min(all_xs), np.max(all_xs)])
ax.set_ylim([np.min(all_ys), np.max(all_ys)])

title_text = ax.text(
    0.5, 1.08, "placeholder", transform=ax.transAxes, ha="center",
    fontsize=15
)
# Prevent display of (duplicate) static figure due to %matplotlib inline
plt.close()

N_frames = len(all_positions)
assert N_frames <= len(all_positions)


def init():
    scat.set_offsets(all_positions[0].T)
    title_text.set_text("")
    return (scat,)


with tqdm(unit=" plots", desc="Plotting", total=N_frames) as t:

    def animate(i):
        scat.set_offsets(all_positions[i].T)
        title_text.set_text("Frame " + str(i))
        t.update()

        return (scat, title_text)

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=N_frames, interval=100,
        blit=True
    )

    js_output = anim.to_jshtml()

HTML(js_output)
```
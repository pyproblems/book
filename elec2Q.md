# Problem 2

This problem revolves around visualising electric field lines using the same techniques used last time to calculate particle motion. You will be given the details of several different charge configurations for which to draw the field lines.

## Background

### Physics

#### Electric Field

As opposed to last problem, we are now interested in the electric field as a result of *all* the charges in the system. Thus, Coulomb's law has to be applied to all charges in the system, leading to the following definition of the electric field:

$$ \mathbf{E} = \sum_{i} k_e \frac{q_i}{|\mathbf{r}_{i}|^2} \mathbf{\hat{r}}_{i} $$

from the definition of the force written in vector form, with $r_{i}$ being the vector from the point of interest to $r_i$.

### Computational Methods

#### SciPy Integration (Adapted from Week 1)

As you learnt previously, using the SciPy [solve_ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html) function will yield significantly better results than a first-order explicit Euler scheme.

In order to use this function, you must rewrite your calculation function to adhere to a function signature akin to that below:

```{code-block} python
def int_fun(t, y, charges):
    """Differential equations for the system of point charges.
    
    Args:
        t (float): Time.
        y (numpy.ndarray): Array containing the state variables.
        charges (numpy.ndarray): (N,) array of charges in C.
    
    Returns:
        iterable: Rates of change of the state variables.
    
    """
    # Your code here ...
```
Note especially that the function **must** take a single float (the 'time') as its first parameter, even if the function never uses this parameter.

The function can then be used like 

```{code-block} python
sol = solve_ivp(
    partial(int_fun, charges=charges),
    [0, 1],
    init_state,
    t_eval=np.linspace(0, 1, 2000),
    rtol=1e-7,
)
# The solutions at different times are contained in sol.y
```

You might have to set the `rtol` parameter to a low number like `1e-7` to achieve realistic results!

As opposed to the first week, this week you might consider using the `solve_ivp` function to integrate _along a field line_. Also note that for this problem, you do **not** need to consider any masses!

## Your Task

You will have to plot field lines for the charge configurations shown below. These will have to be plotted including arrows to indicate the direction of the field. The charge of the different charges should also be apparent.

The range of integration should be large enough to ensure that field lines terminate at charges whenever possible.

### Configurations

Draw field lines for the following scenarios (all positions have units of metres, and charges are given in coulombs):

#### A Simple Configuration

This simple case will depict field lines from two charges at positions (-1, 0) and (1, 0) with charges 1 and -1 respectively.

#### Expanding upon the Simple Configuration

This problem is similar to the one above, in that all the charge are of magnitude 1. The top dipole is a mirrored version of the previous two charges. The charges in this example are positioned at (1, 1), (1, -1), (-1, -1), and (-1, 1) with charges 1, -1, 1, and -1 respectively.

#### A More Complicated Case

Here, 4 charges are placed at the vertices of a square, ie. at (1, 1), (1, -1), (-1, -1), and (-1, 1). Their positive charges are all of magnitude 1.

Another charge is placed in the centre of the square at (0, 0), with charge -2.

### Coding Hints

#### Writing a function to calculate the electric field

Since the electric field is something that we are trying to visualise, it makes sense to write a function to calculate it. For this purpose, you might be able to re-use the function from last week's question.

While the exact implementation is up to you, the function could look something like this:

```{code-block} python
def E(x, y, q, r):
    """Electric field.

    Args:
        x (float): X position(s).
        y (float): Y position(s).
        q (float): Charge(s).
        r (iterable of float): (x, y) position(s) of the point charge(s). If an array is given,
            it should be a (2, N) array where N is the number of point charges.

    Returns:
        float: Electric field vectors at every point in `x` and `y`. The shape of
        this array is the same shape as `x` and `y` with an added initial dimension.

    """
    # Your code here ...
```

While you could write a function to calculate the electric field from a single point charge, it makes sense to vectorise this function so that it accepts multiple charges. Both methods should yield the same result, since the individual electric fields from multiple charges may be added together to yield the combined field.
# Problem

## Background

The relevant concepts and equations are mostly covered in lectures 6–10.
You will be making use of the Huygens–Fresnel principle to simulate wave propagation, interference, and diffraction effects numerically.

## Your Tasks

It is a good idea to create a set of general purpose functions you can use to tackle all of the following problems.
You will end up creating a set of Python functions allowing you to simulate the temporal evolution of waves on a grid of positions.
These waves will emanate from a set of point sources.

Use the same wave speed ($c=1$) and angular velocity ($w=100$) throughout.

### Amplitude Animation

Animate the wave amplitudes (but not their intensities) emanating from a source at $(0, 0)$ over a grid spanning several wavelengths in each direction for 200 frames.
See the Section [Coding Hints](#coding-hints) for an example of code you can use to animate two-dimensional data.
Choose a time interval between the frames that allows you to smoothly visualise the wave evolution, while avoiding aliasing.

### Spatial Coherence

Animate the amplitude evolution of the following point sources, with the same considerations as above:

|$x$   |$y$    |amplitude|
|-----:|------:|--------:|
|0     | 0     | 1       |
|0.05  | 0.05  | -1      |
|-0.05 | -0.05 | 1       |
|0.05  | -0.05 | -1      |
|-0.05 | 0.05  |  1      |

You should be able to observe a transition between the near-field and far-field regimes in your animation.

### Single Slit Diffraction

Using a single slit of width $2 \lambda$, where $\lambda$ is the wavelength (given $c$ and $w$ as above), adapt your functions from the previous tasks in order to recreate the single slit far-field diffraction pattern.
You will need to induce diffraction by placing several point sources within the slit along a single line.

How many point sources are required to create the expected diffraction pattern?

### Triple Slit Diffraction

Using the same slit width as above, simulate the triple slit far-field diffraction pattern using a slit separation of $5\lambda$ (between the centres of the slits).

### Phased Arrays

Phased arrays make use of interference in order to steer a beam of radiation in a particular direction.
Using 40 point sources spaced $0.25\lambda$ apart, adjust the relative phase between each of the point sources to steer the radiation pattern.

What is the relationship between the angle of the (dominant) beam and the relative phase?

### Phase and Amplitude Optimisation

While the previous task may be solved using geometrical considerations, this quickly becomes infeasible when sources are placed at arbitrary positions.
Using the `scipy.optimize` module, determine the phase offsets and amplitudes for each of the 10 sources placed according to:
```python
source_xs, source_ys = np.random.RandomState(0).random((2, 10)) * 0.03
```
in order to maximise the intensity along a $45^{\circ}$ angle from the vertical in the $x$–$y$ plane.

How does the 'best' attainable intensity compare to the intensity resulting from the equally-spaced sources in the previous exercise?

## Coding Hints

You can animate data using the following code, which is meant to be run inside of a **Jupyter Notebook cell**:

```python
%matplotlib inline

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from matplotlib import animation


def animate(
    data, figsize=(10, 8), interval=50, vmin=None, vmax=None,
):
    """2D animation of given data.

    Parameters
    ----------
    data : array-like
        Array containing data to animate, with shape (T, M, N).
    figsize : 2-tuple of int
        Figure size.
    interval : float
        Interval between frames in milliseconds.
    vmin, vmax : float or None
        Colorbar range. If None, inferred from the data.

    Returns
    -------
    animation : HTML representation of the rendered animation.

    """
    fig, ax = plt.subplots(figsize=figsize)
    plt.close(fig)  # So 2 figures don't show up in Jupyter.

    title_text = ax.text(
        0.5, 1.08, "", transform=ax.transAxes, ha="center", fontsize=12
    )

    mesh = ax.pcolorfast(data[0], cmap="RdBu_r", vmin=vmin, vmax=vmax,)

    N_frames = data.shape[0]

    def init():
        mesh.set_data(data[0].T)
        title_text.set_text("")
        return (mesh,)

    def animate(i):
        mesh.set_data(data[i].T)
        title_text.set_text(i)
        return mesh, title_text

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=N_frames, interval=interval, blit=True
    )

    return HTML(anim.to_jshtml())


animate(np.random.random((10, 50, 50)))
```

The final line calls the `animate` function with the data that is intended to be visualised.
This **must** be placed **last** in a Notebook cell in order for the animation to be displayed automatically.
Note that the shape of the array being animated is `(10, 50, 50)`, which means that there are 10 frames (time intervals), and $50\times50$ points in the spatial domain.

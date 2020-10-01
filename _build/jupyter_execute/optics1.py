#!/usr/bin/env python
# coding: utf-8

# # Answers

# ## Setup

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import math
import os
from numbers import Number
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from matplotlib import animation
from scipy.optimize import basinhopping, minimize
from tqdm.notebook import tqdm

mpl.rcParams["animation.embed_limit"] = 500.0

c = 1


# ## Function Definitions

# In[2]:


def fraunhofer_intensity(x, lam, d, a, b, N):
    """Fraunhofer diffraction pattern intensity.

    Parameters
    ----------
    x : float
        Distance from the centre of the diffraction pattern.
    lam : float
        Wavelength.
    d : float
        Distance between the screen and the diffraction slits.
    a : float
        Distance between the diffraction slits.
    b : float
        Diffraction slit width.
    N : int
        Number of slits.

    Returns
    -------
    intensity : float
        The diffraction pattern intensity at `x`.

    """
    arg = np.pi * x / (lam * np.sqrt(x ** 2 + d ** 2))
    arg0 = b * arg
    arg1 = a * arg
    return (np.sin(arg0) ** 2 / arg0 ** 2) * (np.sin(N * arg1) ** 2 / np.sin(arg1) ** 2)


def radial_distance(x, y, x0=0, y0=0):
    """Calculate the distance from a given point.

    Parameters
    ----------
    x : float
        x-coordinate.
    y : float
        y-coordinate.
    x0 : float
        Reference x-coordinate.
    y0 : float
        Reference y-coordinate.

    Returns
    -------
    distance : float
        Distance of the point (`x`, `y`) from the point (`x0`, `y0`).

    """
    return ((x - x0) ** 2 + (y - y0) ** 2) ** 0.5


def e_field(r, e0, w, t, phi=0):
    """Calculate the electric field.

    Parameters
    ----------
    r : array-like
        Distance from the source.
    e0 : float
        Source amplitude.
    w : float
        Angular frequency.
    t : array-like
        Time.
    phi : float
        Phase offset.

    Returns
    -------
    field : array-like
        The electric field from the source with amplitude `e0` at a distance `r` away
        from the source, at time `t`.

    """
    k = w / c

    # Avoid dividing by 0.
    zero_mask = np.isclose(r, 0)
    r[zero_mask] += 1e-6

    if r.ndim == 3 and r.shape[0] == 1:
        # Calculation of E-field for multiple time periods, but with stationary sources.
        old_zero_mask = zero_mask
        zero_mask = np.zeros((t.size, r.shape[-2], r.shape[-1]), dtype=np.bool_)
        zero_mask[:] = old_zero_mask

    # Calculate the electric field.
    field = (e0 / r) * np.cos(k * r - w * t + phi)

    if np.any(zero_mask):
        # Clip radial distance '0s' to observed minima / maxima for valid locations.
        field[zero_mask] = np.clip(
            field[zero_mask], np.min(field[~zero_mask]), np.max(field[~zero_mask])
        )

    return field


def get_centres(x):
    """Calculate the means of consecutive elements in an array."""
    return (x[1:] + x[:-1]) / 2.0


def animate(
    *args,
    figsize=(10, 8),
    interval=50,
    save=False,
    save_file=None,
    vmin=None,
    vmax=None,
    axis=None,
):
    """2D animation of given data.

    Parameters
    ----------
    [xs, ys], data : array-like
        Array containing data to animate, with shape (T, M, N).
    figsize : 2-tuple of int
        Figure size.
    interval : float
        Interval between frames in milliseconds.
    save : bool
        Save the animation as an mp4 file.
    save_file : pathlike, str, or None
        Filename to save the animation in. Only used if `save` is True.
    vmin, vmax : float or None
        Colorbar range. If None, inferred from the data.
    axis : str or None
        For example, if 'square' is given, shapes will not be distorted by the size of
        the axes, i.e. circles will be drawn as circles.

    Returns
    -------
    animation : HTML representation of the rendered animation.
        Only present if `save` is False.

    """
    if len(args) == 1:
        data = args[0]
        initial_plot_args = (data[0].T,)
    else:
        xs, ys, data = args
        initial_plot_args = (xs, ys, data[0].T)

    fig, ax = plt.subplots(figsize=figsize)
    plt.close(fig)  # So 2 figures don't show up in Jupyter.

    title_text = ax.text(
        0.5, 1.08, "", transform=ax.transAxes, ha="center", fontsize=12
    )

    mesh = ax.pcolorfast(*initial_plot_args, cmap="RdBu_r", vmin=vmin, vmax=vmax)

    N_frames = data.shape[0]

    def init():
        mesh.set_data(data[0].T)
        title_text.set_text("")
        return (mesh,)

    if axis is not None:
        ax.axis(axis)

    with tqdm(unit="frame", desc="Rendering", total=N_frames) as tqdm_anim:

        def animate(i):
            mesh.set_data(data[i].T)
            title_text.set_text(i)
            tqdm_anim.update()
            return mesh, title_text

        anim = animation.FuncAnimation(
            fig, animate, init_func=init, frames=N_frames, interval=interval, blit=True
        )

        if save:
            try:
                cwd = os.getcwd()
                if save_file is None:
                    save_dir = cwd
                    filename = "data.mp4"
                else:
                    save_file = Path(save_file)
                    save_dir = save_file.parent
                    filename = save_file.name
                    if filename.suffix.lower() != ".mp4":
                        filename += ".mp4"

                os.chdir(os.path.expanduser(save_dir))
                anim.save(filename, writer=animation.writers["ffmpeg"]())
            finally:
                # Ensure the working directory is not changed.
                os.chdir(cwd)
        else:
            return HTML(anim.to_jshtml())


def huygens_ind_sources(
    N_frames=1,
    dt=1e-4,
    w=100,
    propagation=False,
    xs=np.linspace(0, 1, 200),
    ys=np.linspace(0, 1, 200),
    source_xs=0,
    source_ys=0,
    source_amplitudes=1,
    source_phis=0,
    square=False,
    verbose=True,
):
    """Compute electric field strengths on a grid given a set of sources.

    Parameters
    ----------
    N_frames : int
        Number of frames.
    dt : float
        Time between consecutive frames.
    w : float
        Angular frequency.
    propagation : bool
        If true, execute calculations as if waves are emanating from the given
        sources. Otherwise, compute the steady-state conditions across the entire grid
        given by `xs` and `ys` for all times.
    xs : array-like
        x-coordinates for which to calculate the combined field. These coordinates
        will specify the edges of the resulting bins (see `Returns`).
    ys : array-like
        y-coordinates for which to calculate the combined field. These coordinates
        will specify the edges of the resulting bins (see `Returns`).
    source_xs : array-like
        Source x-coordinates.
    source_ys : array-like
        Source y-coordinates.
    source_amplitudes : array-like
        Source amplitudes.
    source_phis : array-like
        Source phase offset.
    square : bool
        If true, return the amplitudes instead of the field.
    verbose : bool
        If true, output progress information.

    Returns
    -------
    field : array-like with shape (`N_frames`, `xs.size - 1`, `ys.size - 1`)
        Combined field from all the specified sources on a grid given by `xs` and
        `ys` for the times specfied by `N_frames` and `dt`.

    """
    xs = np.asarray(xs)

    if isinstance(ys, Number):
        ys = np.array([ys, ys])
    ys = np.asarray(ys)

    if isinstance(source_xs, Number):
        source_xs = np.array([source_xs])
    source_xs = np.asarray(source_xs)

    if isinstance(source_ys, Number):
        source_ys = np.array([source_ys])
    source_ys = np.asarray(source_ys)

    if isinstance(source_phis, Number):
        source_phis = source_phis * np.arange(source_xs.size)
    source_phis = np.asarray(source_phis)

    if isinstance(source_amplitudes, Number):
        source_amplitudes = source_amplitudes * np.ones(source_xs.size)
    source_amplitudes = np.asarray(source_amplitudes)

    ts = np.linspace(0, (N_frames - 1) * dt, N_frames)
    xc, yc = np.meshgrid(get_centres(xs), get_centres(ys), indexing="ij")
    data = np.zeros((N_frames, xs.size - 1, ys.size - 1))

    # Iterate over sources and calculate their individual contributions to the final field.
    for e0, x0, y0, phi in zip(
        tqdm(source_amplitudes, disable=not verbose), source_xs, source_ys, source_phis
    ):
        r = radial_distance(xc, yc, x0, y0)
        new = e_field(r.reshape(-1, *r.shape), e0, w, ts.reshape(-1, 1, 1), phi=phi)

        if propagation:
            for i, t in enumerate(ts):
                # Create mask based on elapsed time - wave propagation.
                valid = r <= (c * t)
                if np.all(valid):
                    # No need to keep checking, since the wave will keep propagating outwards.
                    break
                new[i][~valid] = 0

        # Add the field from this source to the overall field.
        data += new

    if square:
        return data ** 2.0
    return data


def huygens_interference(
    N_frames=1,
    dt=1e-4,
    w=1000000,
    propagation=False,
    xs=np.linspace(0, 1, 400),
    ys=np.linspace(0, 10, 400),
    x_centre=0,
    N_slits=1,
    N_sources_factor=50,
    slit_separation=0.1,
    slit_width=1e-2,
    square=True,
    phis=0,
    verbose=True,
    row_offsets=0,
):
    """Compute electric field strengths on a grid for an N-slit experiment.

    Parameters
    ----------
    N_frames : int
        Number of frames.
    dt : float
        Time between consecutive frames.
    w : float
        Angular frequency.
    propagation : bool
        If true, execute calculations as if waves are emanating from the given
        sources. Otherwise, compute the steady-state conditions across the entire grid
        given by `xs` and `ys` for all times.
    xs : array-like
        x-coordinates for which to calculate the combined field. These coordinates
        will specify the edges of the resulting bins (see `Returns`).
    ys : array-like
        y-coordinates for which to calculate the combined field. These coordinates
        will specify the edges of the resulting bins (see `Returns`).
    x_centre : float
        Central position of the diffraction slits.
    N_slits : int
        Number of slits.
    N_sources_factor : int
        Number of individual sources per slit.
    slit_separation : float
        Separation between the slits.
    slit_width : float
        Slit widths.
    square : bool
        If true, return the amplitudes instead of the field.
    phis : array-like or float
        Phase offset for each slit. If a single float is given, this will specify the
        phase offset between consecutive slits.
    verbose : bool
        If true, output progress information.
    row_offsets : array-like or float
        If multiple values are given, multiple rows of sources will be constructed,
        each displaced by their respective value in `row_offsets`.

    Returns
    -------
    field : array-like with shape (`N_frames`, `xs.size - 1`, `ys.size - 1`)
        Combined field from all the specified sources on a grid given by `xs` and
        `ys` for the times specfied by `N_frames` and `dt`.

    """
    if isinstance(ys, Number):
        ys = np.array([ys, ys])

    if isinstance(phis, Number):
        phis = phis * np.arange(N_slits)

    if isinstance(row_offsets, Number):
        row_offsets = np.array([row_offsets])

    ys = np.asarray(ys)
    phis = np.asarray(phis)
    row_offsets = np.asarray(row_offsets)

    N_sources = N_sources_factor * N_slits
    min_pos = x_centre - (N_slits - 1) * slit_separation / 2
    max_pos = x_centre + (N_slits - 1) * slit_separation / 2
    slit_positions = np.linspace(min_pos, max_pos, N_slits)
    step_interval = math.ceil(N_sources / N_slits)

    source_offsets = (
        np.linspace(-slit_width / 2, slit_width / 2, math.ceil(N_sources / N_slits))
        if N_sources_factor > 1
        else np.array([0])
    )

    if phis.size == (N_sources * row_offsets.size):
        phis = phis.reshape(-1, row_offsets.size)

    # Iterate over sources and calculate their individual parameters.
    source_xs = []
    source_ys = []
    source_amplitudes = []
    source_phis = []

    for j in tqdm(range(N_sources), disable=not verbose):
        e0 = 1
        x0 = slit_positions[j // step_interval] + source_offsets[j % step_interval]
        phi = phis[j // step_interval]

        if isinstance(phi, Number):
            # Only a single phase offset per row.
            row_phis = [phi] * len(row_offsets)
        else:
            row_phis = phi

        for y0, row_phi in zip(row_offsets, row_phis):
            source_xs.append(x0)
            source_ys.append(y0)
            source_amplitudes.append(e0)
            source_phis.append(row_phi)

    return huygens_ind_sources(
        N_frames=N_frames,
        dt=dt,
        w=w,
        propagation=propagation,
        xs=xs,
        ys=ys,
        source_xs=source_xs,
        source_ys=source_ys,
        source_amplitudes=source_amplitudes,
        source_phis=source_phis,
        square=square,
        verbose=verbose,
    )


# ## Testing the Huygens Functions

# In[3]:


xs = np.linspace(-0.5, 0.5, 500)
ys = np.linspace(-0.5, 0.5, 500)

w = 100
lam = 2 * np.pi * c / w
print("f:", c / lam)

data = huygens_ind_sources(N_frames=150, dt=0.006, w=w, xs=xs, ys=ys, propagation=True)
lim = 6
animate(xs, ys, data, vmin=-lim, vmax=lim, axis="square")


# ## Spatial Coherence

# In[4]:


xs = np.linspace(-0.5, 0.5, 400)
ys = np.linspace(-0.5, 0.5, 400)

w = 100
lam = 2 * np.pi * c / w

print("f:", c / lam)

L = 0.05

data = huygens_ind_sources(
    N_frames=150,
    dt=0.006,
    w=w,
    xs=xs,
    ys=ys,
    propagation=True,
    source_xs=[0, L, -L, L, -L],
    source_ys=[0, L, -L, -L, L],
    source_amplitudes=[1, -1, 1, -1, 1],
)
animate(xs, ys, data, vmin=-8, vmax=8, axis="square")


# ## Single Slit Diffraction

# In[5]:


def plot_far_field(ax, xs, data_sq, lam, ys, slit_separation, slit_width, N_slits):
    centres = get_centres(xs)
    simulated = np.max(data_sq, axis=0)[:, -1]
    ax.plot(centres, simulated, label="simulated", zorder=2)

    fraunhofer = fraunhofer_intensity(
        centres, lam, np.max(ys), slit_separation, slit_width, N_slits
    )
    ax.plot(
        centres,
        np.max(simulated) * fraunhofer / np.max(fraunhofer),
        label="fraunhofer",
        zorder=1,
        linestyle="--",
    )

    ax.legend(loc="best")


# In[6]:


xs = np.linspace(-500, 500, 200)
ys = np.linspace(0, 300, 40)

w = 100
lam = 2 * np.pi * c / w
slit_width = 2 * lam

print("lambda:", lam)
print("a^2 / L:", slit_width ** 2 / (np.max(ys)))
print("Fraunhofer approximation: a^2 / L << lambda")

slit_separation = 10 * lam
N_slits = 1
N_sources_factor = 400

data_sq = huygens_interference(
    N_frames=40,
    dt=3e-3,
    w=w,
    xs=xs,
    ys=ys,
    N_slits=N_slits,
    N_sources_factor=N_sources_factor,
    slit_separation=slit_separation,
    slit_width=slit_width,
)


fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].pcolorfast(xs, ys, np.max(data_sq, axis=0).T, vmin=0, vmax=0.05)
plot_far_field(axes[1], xs, data_sq, lam, ys, slit_separation, slit_width, N_slits)

# Recalculate just the final pattern.
xs2 = np.linspace(np.min(xs), np.max(xs), 1000)

data_sq = huygens_interference(
    N_frames=40,
    dt=3e-3,
    w=w,
    xs=xs2,
    ys=np.max(ys),
    N_slits=N_slits,
    N_sources_factor=N_sources_factor,
    slit_separation=slit_separation,
    slit_width=slit_width,
)

fig, ax = plt.subplots(figsize=(14, 6))
plot_far_field(ax, xs2, data_sq, lam, ys, slit_separation, slit_width, N_slits)
_ = ax.set_ylim(0, ax.get_ylim()[1] / 5)


# ## Triple Slit Diffraction

# In[7]:


xs = np.linspace(-500, 500, 200)
ys = np.linspace(0, 300, 40)

w = 100
lam = 2 * np.pi * c / w
slit_width = 2 * lam

print("lambda:", lam)
print("a^2 / L:", slit_width ** 2 / (np.max(ys)))
print("Fraunhofer approximation: a^2 / L << lambda")

slit_separation = 10 * lam
N_slits = 3
N_sources_factor = 400

data_sq = huygens_interference(
    N_frames=40,
    dt=3e-3,
    w=w,
    xs=xs,
    ys=ys,
    N_slits=N_slits,
    N_sources_factor=N_sources_factor,
    slit_separation=slit_separation,
    slit_width=slit_width,
)


fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].pcolorfast(xs, ys, np.max(data_sq, axis=0).T, vmin=0, vmax=0.05)
plot_far_field(axes[1], xs, data_sq, lam, ys, slit_separation, slit_width, N_slits)

# Recalculate just the final pattern.
xs2 = np.linspace(np.min(xs), np.max(xs), 1000)

data_sq = huygens_interference(
    N_frames=40,
    dt=3e-3,
    w=w,
    xs=xs2,
    ys=np.max(ys),
    N_slits=N_slits,
    N_sources_factor=N_sources_factor,
    slit_separation=slit_separation,
    slit_width=slit_width,
)

fig, ax = plt.subplots(figsize=(14, 6))
plot_far_field(ax, xs2, data_sq, lam, ys, slit_separation, slit_width, N_slits)
_ = ax.set_ylim(0, ax.get_ylim()[1] / 5)


# ## Phased Array - Beamforming
# 
# The direction of the dominant beam is determined by the phase offset between each of the sources.

# In[8]:


xs = np.linspace(-300, 300, 200)
ys = np.linspace(0, 1000, 200)

w = 100
lam = 2 * np.pi * c / w
slit_width = 2 * lam

data_sq = huygens_interference(
    N_frames=20,
    dt=3e-3,
    w=w,
    xs=xs,
    ys=ys,
    N_slits=40,
    N_sources_factor=1,
    slit_separation=0.25 * lam,
    slit_width=0,
    phis=-0.35 * np.pi,
)

fig, ax = plt.subplots(figsize=(10, 10))
ax.pcolorfast(xs, ys, np.max(data_sq, axis=0).T, vmin=0, vmax=0.01)


# In[9]:


xs = np.linspace(-300, 300, 200)
ys = np.linspace(0, 1000, 200)

w = 100
lam = 2 * np.pi * c / w
slit_width = 2 * lam

data_sq = huygens_interference(
    N_frames=20,
    dt=3e-3,
    w=w,
    xs=xs,
    ys=ys,
    N_slits=40,
    N_sources_factor=1,
    slit_separation=0.25 * lam,
    slit_width=0,
    phis=0.1 * np.pi,
)

fig, ax = plt.subplots(figsize=(10, 10))
ax.pcolorfast(xs, ys, np.max(data_sq, axis=0).T, vmin=0, vmax=0.01)


# ## Phased Array - Phase Offset and Amplitude Opimisation
# 
# Numerical optimisation is used to derive ideal phase offsets to maximise the beam intensity at a single chosen point.

# In[10]:


# Define the coordinate grid.
L = 20
xs = np.linspace(-L, L, 200)
ys = np.linspace(-L, L, 200)

# Define the coordinates of a point at which to maximise signal intensity.
xt, yt = (L, L)

# Define common parameters.
w = 100
lam = 2 * np.pi * c / w

n_points = 10

source_xs, source_ys = np.random.RandomState(0).random((2, n_points))

# Define a function which will take the phase offsets as an input.
# This function will then be minised in order to maximise the output intensity.


def point_intensity(args):
    source_phis = args[:n_points]
    source_amplitudes = args[n_points : 2 * n_points]

    source_amplitudes = np.abs(source_amplitudes)
    source_amplitudes /= np.max(source_amplitudes)

    data_sq = huygens_ind_sources(
        w=w,
        N_frames=40,
        dt=2e-3,
        xs=np.array([xt, xt]),
        ys=np.array([yt, yt]),
        source_xs=source_xs,
        source_ys=source_ys,
        source_phis=source_phis,
        source_amplitudes=source_amplitudes,
        verbose=False,
    )
    return -np.max(data_sq)


# Optimise phase offsets.

x0 = np.append(np.zeros(n_points), np.ones(n_points))
res = basinhopping(point_intensity, x0, disp=False,)

opt_phis = res.x[:n_points]
opt_amplitudes = res.x[n_points : 2 * n_points]
opt_amplitudes = np.abs(opt_amplitudes)
opt_amplitudes /= np.max(opt_amplitudes)


# In[11]:


data = huygens_ind_sources(
    w=w,
    N_frames=20,
    dt=2e-3,
    xs=xs,
    ys=ys,
    source_xs=source_xs,
    source_ys=source_ys,
    source_phis=opt_phis,
    source_amplitudes=opt_amplitudes,
    square=False,
)

fig, ax = plt.subplots(figsize=(10, 10))
plot_data = np.max(data, axis=0).T
plot_data /= np.sum(plot_data)

lim = 0.002
ax.pcolorfast(xs, ys, plot_data, vmin=0, vmax=lim, cmap="viridis")

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(opt_phis % (2 * np.pi), label="phase")
ax2 = ax.twinx()
ax2.plot(opt_amplitudes, label="amplitudes", c="C1")
ax.legend(loc="upper right")
_ = ax2.legend(loc="upper left")


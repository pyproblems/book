# Problem 1 - Answers

## Imports and Setup

Matplotlib inline plotting will set the matplotlib backend to ensure that the plots will not appear in a separate window, but rather in the notebook itself. 
Of course plots can always be saved to a separate file using the `savefig` command or shown in a separate window by changing the [matplotlib backend](https://matplotlib.org/3.1.1/tutorials/introductory/usage.html#backends) (advanced!).

These imported packages are used throughout the Jupyter Notebook.
The [matplotlib](https://matplotlib.org/) package is used for plotting, while [NumPy](https://numpy.org/) is used for underlying numerical calculations.
[SciPy](https://www.scipy.org/) provides multiple advanced routines, such as routines for solving differential equations which are used in this notebook.
[Tqdm](https://tqdm.github.io/) is used to record the elapsed time for computations and predict how long a given part of the program is expected to take to run.
The electrostatics module defines simple routines to calculate electric fields for different arrangements of charges.
It also defines a Timer class which can be used to time isolated parts of code.

%matplotlib inline
import time
from functools import partial

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from matplotlib import animation
from scipy.integrate import solve_ivp
from tqdm import tqdm

from electrostatics import E, Timer, k_e

mpl.rc("figure", figsize=(10, 10))
mpl.rc("font", size=16)
mpl.rc("animation", embed_limit=40)

## SciPy Solver Function

The function below will be used in conjunction with the SciPy solver to solve for the particles' trajectories.

def int_fun(t, y, masses, charges):
    """Differential equations for the system of point charges.
    
    Args:
        t (float): Time.
        y (numpy.ndarray): Array containing the state variables,
            (x_1_x, x_1_y, v_1_x, v_1_y, ...), the position and velocity components
            of all the charges.
        masses (numpy.ndarray): (N,) array of masses in kg.
        charges (numpy.ndarray): (N,) array of charges in C.
    
    Returns:
        iterable: Rates of change of the state variables (v_1_x, v_1_y, a_1_x, a_1_y, ...).
    
    """
    # Reshape so that the first 2 columns contain positions, and the last 2
    # columns contain velocities.
    state = y.reshape(-1, 4)
    N = len(state)
    ndim = 2

    accelerations = []

    for i in range(N):
        accelerations.append(
            charges[i]
            * np.sum(
                E(
                    *state[i, :2],
                    np.array([charges[j] for j in range(N) if j != i]),
                    np.hstack(
                        [state[j, :2].reshape(ndim, 1) for j in range(N) if j != i]
                    )
                ),
                axis=1,
            ).reshape(ndim)
            / masses[i]
        )

    out = []
    for i in range(N):
        out.extend(state[i, 2:4])
        out.extend(accelerations[i])

    return out

## Euler Method

The function below contains all the logic necessary for the Euler method.

The positions are first updated using the known velocities.
Then, the accelerations of all particles are determined using the electric field at that point in time.
Finally, new velocities are calculated using these accelerations.
When the function is next called, these new positions and velocities are then supplied, thereby advancing the solution by one step.

def euler_step(positions, velocities, masses, charges, h):
    """Get new positions and velocities using the Euler method.
    
    Args:
        positions (numpy.ndarray): (2, N) array of (x, y) positions in m.
        velocities (numpy.ndarray): (2, N) array of (x, y) velocities m/s.
        masses (numpy.ndarray): (N,) array of masses in kg.
        charges (numpy.ndarray): (N,) array of charges in C.
        h (float): Temporal step size in seconds.
        
    Returns:
        numpy.ndarray, numpy.ndarray: New positions and new velocities after the update.
        
    """
    N = positions.shape[1]
    ndim = positions.shape[0]
    new_positions = positions + h * velocities
    # Calculate accelerations from the Lorentz force law.
    accelerations = []
    for i in range(N):
        accelerations.append(
            charges[i]
            * np.sum(
                E(
                    *positions[:, i],
                    np.array([charges[j] for j in range(N) if j != i]),
                    np.hstack(
                        [positions[:, j].reshape(ndim, 1) for j in range(N) if j != i]
                    )
                ),
                axis=1,
            ).reshape(ndim, 1)
            / masses[i]
        )

    accelerations = np.hstack(accelerations)
    new_velocities = velocities + h * accelerations

    return new_positions, new_velocities

def get_initial(verbose=False):
    """Get initial parameters.
    
    Args:
        verbose (bool): If True, print information regarding the initial 
            parameters.
            
    Returns:
        iterable: The positions, velocities, masses, and charges are returned.
    
    """
    positions = np.array([[2, 1, 3], [1, 3, 2]], dtype=np.float64)
    velocities = np.zeros_like(positions)
    masses = np.array([1, 3, 2], dtype=np.float64)
    charges = np.array([1, -4, 2], dtype=np.float64)

    if verbose:
        for i in range(positions.shape[1]):
            print(f"Particle {i}")
            print(positions[:, i])
            print(masses[i])
            print(charges[i])
    return positions, velocities, masses, charges

## Using the Euler Method

positions, velocities, masses, charges = get_initial()

# Define the step size.
h = 0.05e-6

all_positions = []

# Run the simulation for 2000 steps.
for i in range(2000):
    positions, velocities = euler_step(positions, velocities, masses, charges, h)
    all_positions.append(positions)

fig, ax = plt.subplots()
scat = ax.scatter(*all_positions[0], c=charges)

all_xs = [x for step_positions in all_positions for x in step_positions[0]]
all_ys = [y for step_positions in all_positions for y in step_positions[1]]

ax.set_xlim([np.min(all_xs), np.max(all_xs)])
ax.set_ylim([np.min(all_ys), np.max(all_ys)])

title_text = ax.text(
    0.5, 1.08, "placeholder", transform=ax.transAxes, ha="center", fontsize=15
)
plt.close()  # Prevent display of (duplicate) static figure due to %matplotlib inline

N_frames = len(all_positions)
assert N_frames <= len(all_positions)


def init():
    scat.set_offsets(all_positions[0].T)
    title_text.set_text("")
    return (scat,)

def animate(i):
    scat.set_offsets(all_positions[i].T)
    title_text.set_text(f"Euler - Frame {i:04d}")

    return (scat, title_text)

anim = animation.FuncAnimation(
    fig, animate, init_func=init, frames=N_frames, interval=100, blit=True
)

print('This will take a little while...')

js_output = anim.to_jshtml()

HTML(js_output)

## Using the Scipy Solver

positions, velocities, masses, charges = get_initial()

init_state = []
for i in range(positions.shape[1]):
    # Add x, y coordinates.
    init_state.extend(positions[:, i])
    # Add x, y velocities.
    init_state.extend(velocities[:, i])

sol = solve_ivp(
    partial(int_fun, masses=masses, charges=charges),
    [0, 1e-4],
    init_state,
    t_eval=np.linspace(0, 1e-4, 2000),
    rtol=1e-7,
)

fig, ax = plt.subplots()
scat = ax.scatter(*all_positions[0], c=charges)

all_xs = sol.y[::4]
all_ys = sol.y[1::4]

ax.set_xlim([np.min(all_xs), np.max(all_xs)])
ax.set_ylim([np.min(all_ys), np.max(all_ys)])

title_text = ax.text(
    0.5, 1.08, "placeholder", transform=ax.transAxes, ha="center", fontsize=15
)
plt.close()  # Prevent display of (duplicate) static figure due to %matplotlib inline

N_frames = sol.y.shape[1]

all_positions = []
for i in range(N_frames):
    all_positions.append(np.array([sol.y[::4, i], sol.y[1::4, i]]))


def init():
    scat.set_offsets(all_positions[0].T)
    title_text.set_text("")
    return (scat,)


def animate(i):
    scat.set_offsets(all_positions[i].T)
    title_text.set_text(f"SciPy - Frame {i:04d}")

    return (scat, title_text)

anim = animation.FuncAnimation(
    fig, animate, init_func=init, frames=N_frames, interval=100, blit=True
)

print('This will take a little while...')

js_output = anim.to_jshtml()

HTML(js_output)

## Comparing Euler Method with Scipy solver

Here, the evolution of the x-position of particle 1 is shown for both cases.
Even though the Euler method takes significantly longer than the SciPy solver to find the solution, it only performs similarly when the accelerations are low.
Upon encountering a 'dramatic' event, such as a close encounter between charges, the Euler method (with the currently selected step size) fails to capture the expected motion.

# Define basic solver parameters.

N = int(1e5)
T_end = 1e-4
h = T_end / N

times = np.linspace(0, h * N, N)

# First calculate the Euler solution.
positions, velocities, masses, charges = get_initial()

all_positions = []

with Timer("Euler"):
    for i in range(N):
        positions, velocities = euler_step(positions, velocities, masses, charges, h)
        all_positions.append(positions)

euler_xs = (
    np.array([x for step_positions in all_positions for x in step_positions[0]])
    .reshape(N, -1)
    .T
)

# Then calculate the Scipy (RK) solution.
positions, velocities, masses, charges = get_initial()

init_state = []
for i in range(positions.shape[1]):
    # Add x, y coordinates.
    init_state.extend(positions[:, i])
    # Add x, y velocities.
    init_state.extend(velocities[:, i])

with Timer("RK Scipy"):
    sol = solve_ivp(
        partial(int_fun, masses=masses, charges=charges),
        [0, h * N],
        init_state,
        t_eval=times,
        rtol=1e-7,
    )

rk_xs = sol.y[::4]

# Visualise the difference between the two solutions.
plt.figure()
plt.plot(times, euler_xs[0], label="Euler")
plt.plot(times, rk_xs[0], label="SciPy (RK)")
plt.ylabel("x")
plt.xlabel("t")
plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
_ = plt.legend(loc="best")

## Plotting the Energy of System

Both the electrostatic potential energy of the system and the sum of all kinetic energies is plotted.
It can be seen that their sum is constant, up to very small fluctuations ($\lt 0.01\%$) related to the discretisation error.

The SciPy (RK) solution `sol.y` from the previous cell is used here throughout.

def pot_energy(positions, charges):
    """Calculate the electrostatic potential energy of the system.
    
    Args:
        positions ((2, N) numpy.ndarray): Array containing the charge positions.
        charges ((N,) numpy.ndarray): Array containing the charges.
    
    """
    energy = 0
    for (i, (position_i, charge_i)) in enumerate(zip(positions.T, charges)):
        potential = 0
        for (j, (position_j, charge_j)) in enumerate(zip(positions.T, charges)):
            if i != j:
                potential += charge_j / np.linalg.norm(position_i - position_j)
        energy += 0.5 * k_e * charge_i * potential
    return energy


xs = sol.y[::4]
ys = sol.y[1::4]
pos_arr = np.vstack((xs[np.newaxis], ys[np.newaxis]))

vxs = sol.y[2::4]
vys = sol.y[3::4]

pot_energies = []
for i in range(pos_arr.shape[-1]):
    positions = pos_arr[..., i]
    pot_energies.append(pot_energy(positions, charges))

pot_energies = np.array(pot_energies)

kin_energies = 0.5 * np.sum(
    masses.reshape(3, 1)
    * np.sum(np.vstack((vxs[np.newaxis], vys[np.newaxis])) ** 2, axis=0),
    axis=0,
)

plt.figure()
plt.plot(pot_energies / np.max(np.abs(pot_energies)), label="Potential Energy")
plt.plot(kin_energies / np.max(np.abs(kin_energies)), label="Kinetic Energy", c="C1")
plt.suptitle("Normalised Energies")
plt.xlabel("t")
plt.ylabel("E")
plt.legend(loc="best")
plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
plt.show()

plt.figure()
energy_sum = kin_energies + pot_energies
plt.plot(energy_sum / np.max(np.abs(energy_sum)))
plt.suptitle("Norm. Kinetic + Potential Energy")
plt.xlabel("t")
plt.ylabel("E")
plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
plt.show()

### Inspecting the System at Particular Points

We can visualise the state of the system at interesting points, such as when it possesses the minimum or maximum potential and kinetic energies.
As expected, we can verify that when the system has minimum potential energy, it has maximum kinetic energy and vice versa (see the `assert` statements below).

norm_charges = charges - np.min(charges)
norm_charges /= np.max(norm_charges)
cmap = mpl.cm.get_cmap()
colors = [cmap(norm_charge) for norm_charge in norm_charges]

assert np.argmin(pot_energies) == np.argmax(kin_energies)
assert np.argmax(pot_energies) == np.argmin(kin_energies)

for index, title, ylims in zip(
    (np.argmin(pot_energies), np.argmax(pot_energies)),
    ("Min Potential, Max Kinetic", "Max Potential, Min Kinetic"),
    ((2.215, 2.4), (0.8, 3.2)),
):
    plt.figure()
    for pos_vals, charge, color in zip(pos_arr[..., index].T, charges, colors):
        plt.plot(*pos_vals, marker="o", label=charge, linestyle="", c=color)

    plt.xlim(0.8, 3.2)
    plt.ylim(*ylims)
    plt.suptitle(f"{title} (Frame {index})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
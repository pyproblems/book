# Problem

**Simulation of a neutrino beam**

````{margin}
```{note}
Google italicised terms to learn more!
```
````

Long-baseline neutrino experiments study the nature of neutrinos, and in particular *neutrino oscillations*, in detail by generating neutrino beams in accelerator facilities. The success of experiments like *T2K*, *NO$\nu$A*, and *MiniBooNE* relies on the correct modeling of the neutrino beam. Accurate simulations which make use of random generation techniques, also known as *Monte Carlo methods*, are an essential element of these studies. 

Neutrino beams are typically produced by accelerating protons to high energies against a fixed target (e.g. graphite). The highly energetic interactions between the protons and the carbon nuclei produce secondary particles, principally *mesons*, such as *pions* $\pi$ and *kaons* $K$. These are short-lived particles which produce neutrinos in their decays; the most probable decays are:

$$ \pi^+ \rightarrow \nu_\mu + \mu^+ $$

$$ K^+ \rightarrow \nu_\mu + \mu^+ $$

In this Python exercise we will study the neutrino beams produced by the $\pi$ and $K$ decays. Notions of relativistic kinematics and basic Python skills are required, and we will introduce simple simulation techniques along the way. We will refer to $\pi$, $K$ as parent particles and $\mu$, $\nu$ as product particles. You can use the approximations:

$$ m_\pi = 140\mathrm{MeV} $$

$$ m_K = 490\mathrm{MeV} $$

$$ m_\mu = 105\mathrm{MeV} $$

$$ m_\nu = 0\mathrm{MeV} $$

or look up the values yourself. 

## 1. Parent particle rest frame

**Focus on $\nu_\mu$ production in the parent particle rest frame.**

1.1 Write a function which calculates the energy of particle c in a generic 2-body decay $a \rightarrow b + c$ with masses $m_a, m_b, m_c.$

1.2 Use it to calculate the Energy of the $\nu_\mu$ in both cases.

1.3 Write a function to calculate the 3-momentum of a generic particle $c$ in the same 2-body decay with angle $\theta$ w.r.t. the $z$-axis and azimuthal angle $\phi$.

```{admonition} Hint
:class: dropdown, tip
Try using *numpy arrays*
```

1.4 Use the functions above to calculate the 3-momentum of $\nu$ in both decays, assuming the decay happens on the plane $x = 0$, with a $\theta$ of your choice.

1.5 Write a function that takes the energy and 3-momentum of a particle $c$ in input and returns the 4-momentum.

```{admonition} Hint
:class: dropdown, tip
If you're using numpy arrays, look up the function `insert`
```

## 2. Lab frame

**Assume the $\pi$ and $K$ are travelling in the lab frame with Energy 1GeV in direction $z$.**

2.1 Write a function which returns the value of $\beta = v/c$ in the lab frame, given the mass and energy of the particle.

2.2 Write a function which allows you to boost a 4-momentum along the positive $z$-axis.

2.3  Calculate the value of $\beta$ of the lab frame as seen by the rest frame using 2.1.

```{note}
Careful with the signs!
```

2.4 Calculate the 4-momenta of the neutrinos boosted into the lab frame, assuming 1.5.

```{admonition} Hint
:class: dropdown, tip
A quick check that the boost is correct is to verify the 4-momentum squared is still equal to the mass. Why is this?
```

2.5 What are the maximum and minimum values of the energy the product neutrino can have, in the lab frame? 

```{admonition} Hint
:class: dropdown, tip
Think of theta.
```

## 3. Energy Distributions

**Let's generalise what you have found for different angles $\theta$. Let rf refer to the rest frame and lf to the lab frame. What are the allowed values of $\theta_{rf}$ in the rest frame in this decay?**

3.1 Guided exercise: create a plot of the distribution of energies in the lab frame of the product neutrinos as a function of $\theta$, assuming the parent particle has energy 1GeV in the lab frame. Also look at the histogram of the energies.

a) You can approach this problem in several ways; a natural one is to pick the angle $\theta$ as a random variable many times, until you have a clear picture of the distribution. To do this import `numpy` and use `np.random.rand()` : `theta = np.random.rand()` in a `while` or `for` loop, appending each value to a list. Try running, and make sure you understand, the below code before pressing on. 

```{code-block} python
import numpy as np
import matplotlib.pyplot as plt
## check what the funciton rand() does:

# np.random.rand() will return a random number between 0 and 1
print(np.random.rand())

# adding an amplitude A in front will return a random number between 0 and A, e.g.: 
print(np.pi * np.random.rand())
# returns a random number between 0 and PI

## create an empty list to store your random numbers in:
random = []

## run a for loop which creates random numbers with A = np.pi and appends each new value to the list above
for idx in range(1000):
    random.append(np.pi * np.random.rand())

## plot the output list
import matplotlib.pyplot as plt

plt.plot(random)
plt.show()

## make a histogram of the list

# the function plt.hist() will make a histogram of the argument; you may select the number of bins
plt.hist(random, bins = 100)
plt.show()
# as expected, the histogram looks like a flat distribution
```

b) Write a loop which will sample $\theta$ in its range with `rand()`, saving each value of $\theta$ and the associated value of `energy_nu_lab` for both decays.

c) Create plots for both decays of `energy_nu_lab` as a function of $\theta$. What function is it? 

```{admonition} Hint
:class: dropdown, tip
Use `plt.scatter()` to plot single dots, as opposed to `plt.plot()` which will connect dots with a line!
```

d) Create plots for both decays of the histograms of the `energy_nu_lab`.

3.2 Guided exercise: create a plot of the distribution histogram of energies in the lab frame of the product neutrinos, assuming the parent particle has a log-normal momentum distribution in the lab frame. We can assume the momenta of the $\pi$ and $K$ beams follow the log-normal distribution, and that there are roughly 10 more $\pi$ decays than $K$ decays in the beam. 

a) Familiarise yourself with the log-normal distribution function.

```{code-block} python
def generate_log_normal(x_min,x_max,mu,sigma):
    x = np.random.lognormal(mu,sigma)
    if x <= x_max and x >= x_min:
        return x
    else:
        return generate_log_normal(x_min,x_max,mu,sigma)


# use these values of sigma and mu throughout
sigma = 0.5
mu = 0.25

# recover the min and max values of the energy from your calculations above
x_min = 0. #GeV
x_max = 6. #GeV

# set up the run
distrib_x = []
n_samples = 1000

# pick values from the distribution
for idx in range(n_samples):
    distrib_x.append(generate_log_normal(x_min,x_max,mu,sigma))
    
plt.plot(distrib_x)
plt.show()
plt.close()

plt.hist(distrib_x, bins = 100)
plt.show()
```

b) Repeat exercise in 3.1, now drawing the energy of the parent particles from the distribution at each iteration in the loop, and saving the values of `energy_nu_lab` each time. Remember to still keep the value of $\theta$ random! Finally, plot the histogram of the sum of the energies of the product neutrinos. 

```{note}
You only need to run a single 'for' loop - but you may want to run it over a lot of events so that you sample the distributions of both theta and the energies.
```

c) You run the experiment and you get the histogram in b), which includes all the product neutrinos. How could you differentiate between the $\pi$ neutrinos and the $K$ neutrinos? Replot the histogram for b) separating the $K$ and $\pi$ neutrinos explicitly, and point out the differences.

```{admonition} Hint
:class: dropdown, tip
You may need to use different bin numbers for the two different neutrino beams, or equivalently redefine the range of the histogram, as the number of events is different!
```

For reference, you should get a plot which looks like:

```{figure} images/Relativity_Y1.jpg
---
height: 500px
name: Relativity_Y1
---
Predicted neutrino flux of the MiniBooNE experiment from https://arxiv.org/pdf/0806.1449.pdf
```
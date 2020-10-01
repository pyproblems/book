# Problem 2

**Investigating binary collision cross sections**

Use the `collide` function defined in Problem 1, with the same set of physical and simulation parameters. Amend this function so that instead of returning the alpha trajectories, it instead returns the cosine of the scattering angle (the angle between the initial and final velocities of the particle). 

Hint: Using the numpy `vectorize` decorator on the collide function will make the next steps easier, as you will be able to pass a vector of impact parameters as an argument and get a vector of scattering angles returned. An example of this:

```{code-block} python
@np.vectorize
def my_func(x):
    if x < 0.0:
    	return 0.0
    else:
    	return np.sqrt(x)

my_func([-9,-4,-1,0,1,4,9])
```
will return `array([0., 0., 0., 0., 1., 2., 3.])`. 

## Hard Sphere Scattering

a) Use the `collide` function to evaluate the cosine of the scattering angle that results from simulating hard sphere scattering across a range of impact parameters. Plot the scattering angle as a function of impact parameter.

b) One often considers the differential cross section $\frac{d\sigma}{d\Omega}$, relating the differential solid angle $d\Omega$ into which particles in the differential cross section $d\sigma$ are scattered. This can clearly be related to our known parameters: the impact parameter $b$ and scattering angle $\theta$. Take as given the relation for a radially symmetric scattering potential:

$$ \frac{d\sigma}{d\Omega} = \frac{b}{\sin\theta}\left|\frac{db}{d\theta}\right| $$

Plot the differential cross section for hard sphere scattering against the cosine of the scattering angle. You will know from your notes that this kind of scattering should be isotropic - do you get a uniform distribution? If not, can you convince yourself that any departure from uniformity is a numerical artefact? Try fiddling with simulation parameters and note the effect.

## Rutherford Scattering

a) Repeat the procedure you followed for hard sphere scattering, for the case of Rutherford scattering. Produce plots of the scattering angle as a function of impact parameter and of the differential cross section for hard sphere scattering against the cosine of the scattering angle. 

A theoretical derivation of the differential cross section for Rutherford scattering gives:

$$ \frac{d\sigma}{d\Omega} = \left(\frac{Qq}{8\pi\epsilon_0mv_0^2\sin^2(\theta/2)}\right)^2 $$

How does this compare to your numerical results?

## Mixed Scattering

Repeat the above procedure, for the case of a charged hard sphere. Use a range of sphere radii (such as $1.5\times10^{-13}$m, $3\times10^{-13}$m, and $6\times10^{-13}$m) and each case produce plots of the scattering angle as a function of impact parameter and of the differential cross section for hard sphere scattering against the cosine of the scattering angle. In each case the sphere should have charge of a Gold nucleus. Compare the differential cross sections to the theoretical result for Rutherford scattering. What do you find?
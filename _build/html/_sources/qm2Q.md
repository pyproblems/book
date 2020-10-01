# Problem 2

**Time independent perturbation theory**

In your problem sheets you have calculated the first order corrections to the ground state energy and eigenfunction of a particle in a perturbed infinite square well. In this question you will expand this analysis to higher order.

The potential for this perturbed infinite square well is given by

$V = \infty$ for $x < 0$,&nbsp; &nbsp; &nbsp;$V = \delta$ for $0 < x < a$,&nbsp; &nbsp; &nbsp;$V = 0$ for $a < x < L$,&nbsp; &nbsp; &nbsp;$V = \infty$ for $x > L$.

For the rest of this question assume an electron is confined in the perturbed potential well with  $L = 1\times10^{-10}$ m, $a = 5\times10^{-11}$ m and $\delta = 1\times10^{-19}$ J.

1. Verify that $\delta$ << E1.


2. Calculate the first order shift in the energy of the ground state and compare this to the value you found by hand.

```{admonition} Hint
:class: dropdown, tip
You can use `scipy.interpolate.interp1d` to turn a list of x and y coordinates into a function which you can then integrate using `scipy.integrate.quad`.
```


3. Find the perturbed ground state wavefunction. Compare the calculated $a_{12}$ coefficient to the value you found by hand. Plot the unperturbed and perturbed wavefunctions and the difference between them.

```{note}
Due to slight differences in the way we have set up the problem, the coefficient will differ by a minus sign. Can you explain why?
```


4. By repeatedly applying the functions you have just created, calculate the higher order approximations for the ground state energy. How many iterations are required for the size of the energy shift to halve compared to the first iteration?
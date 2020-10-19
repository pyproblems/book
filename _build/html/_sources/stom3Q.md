# Problem 3

**Chi-square Tests and Parameter Estimation**

The chi-square test is used to test whether observed counts agree with a proposed distribution.
The test statistic is $X = \sum (O - E)^2 / E,$ where $E$ are the expected and $O$ the observed counts, and the sum is taken over all bins.
This is equivalent (given certain conditions) to a chi-square distribution with $k$ - 1 degrees of freedom, where $k$ is the number of bins.
To carry out the test, $X$ is therefore compared to the quantile of the corresponding chi-square distribution dictated by the chosen significance level, $\alpha$.
Generally, $E > 20$ should be satisfied for all bins in order to carry out this test.
While a variety of binning methods are used in practice, you will be using bins with _equal probabilities_ here.
This implies that the bins will, in general, have unequal widths.

## Python packages

While you are free to use any approach you like, the following may come in handy:
 - `chisquare` and `norm` from the `scipy.stats` sub-package
 - `fmin` from `scipy.optimize`

## Your tasks

 1. Generate a random sample from the normal distribution with $\mu=0$ and $\sigma=1$, i.e. $\mathcal{N}(0, 1)$.
 On the same plot, show the expected pdf in addition to a histogram of the generated sample.
 2. From $\mathcal{N}(0, 1)$, determine 11 bin edges (for 10 bins) which each contain equal probabilities, encompassing $[-\inf, +\inf]$.
 Using the chi-square goodness-of-fit test, determine the p-value of a random sample of $\mathcal{N}(0, 1)$, binned according to the aforementioned bin edges.
 Here, we are trying to determine if the sample originated from $\mathcal{N}(0, 1)$ or not.
 Repeat this process $>100$ times until you are satisfied with the results.
 Plot a histogram of all p-values.
 For the $0.10$, $0.05$, and $0.01$ significance levels, calculate the fraction of samples with p-values below $\alpha$.
 Do the significance levels and your calculated fractions agree?
 3. Next, the above will be expanded using maximum likelihood parameter estimation.
    1. Write a function which, given $\mu$, $\sigma$, and an array of observations, returns the value of the (log) likelihood (up to a constant) for the normal distribution.
    2. Test the above function by generating a random sample from $\mathcal{N}(0, 1)$, and evaluating the function using this sample along with a range of values for $\mu$ and $\sigma$ on a grid.
    3. Plot a contour plot of the resulting (log) likelihood values. This should allow you to visually verify that the maximum of the function is located at the expected values $\mu=0$, $\sigma=1$.
    4. Carry out explicit maximisation of the likelihood function using `fmin` and your own function, and verify that the maximum is indeed at the expected location.

 4. Carrying out parameter estimation changes the null hypothesis, $H_0$, being tested—it is now a composite null hypothesis.
 We are now trying to determine if the sample originates from _any_ normal distribution, instead of just $\mathcal{N}(0, 1)$.
 The calculations from 2. will be repeated here with the additional step of parameter estimation, as above.
 Thus, instead of fixed bin edges for every trial, these will now depend on the drawn samples and the resulting estimates for $\mu$ and $\sigma$.
 Thereafter, the chi-square statistic and p-value is calculated as before.
 Again, calculate the fraction of rejected samples for various significance levels.
 This time, the degrees of freedom used for the chi-square test will need to be varied.
 Carry out the above for $k-1$, $k-2$, $k-3$, and $k-4$ degrees of freedom, where $k$ is the number of bins.
 Which number of degrees of freedom is appropriate here, and is this supported by your data?
 5. Finally, carry out the parameter estimation in the last exercise using chi-square minimisation instead of maximising the likelihood function.
    1. First define a function which will be minimized using `fmin`.
    It should take the mean and standard deviation of the normal distribution, bin edges, and observed counts as arguments, and return the corresponding chi-square value.
    This will involve calculating the expected counts in the given bins resulting from the given $\mu$ and $\sigma$.

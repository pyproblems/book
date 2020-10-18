# The CLT applied to sums of dice

Using the functions created in the first exercise, you will estimate the sum of of particular die combinations, comparing this to the calculated analytical results.

## Your tasks

 1. Make your functions from the first exercise available to the current notebook (or otherwise), so that they are reusable.
 2. Sample from the distribution of the sum of 10 6-sided dice and plot a bar chart of the probabilities.
 3. Compute the analytical expected probability for each of the bins above, and compare the sampled results to the expected results using a grouped bar chart.
 Also sample the normal distribution with the expected mean and standard deviation of the sum at the bin centres and add this to the grouped bar chart (after normalisation of the probabilities, ensuring they add up to 1).
 4. Write a function that calculates the expected probabilities in each bin using the normal distribution and a continuity correction: instead of sampling the normal distribution's pdf at the bin centres, calculate the probability contained within $\pm 0.5$ of each bin (or to $\pm \infty$ at the edges of the dstribution).
 5. Compare the results of the above function, for the case in Task 3, adding this to the grouped bar chart.
 6. In preparation for the next task, write a function that takes bin edges, the corresponding probabilities for each bin, and a probability threshold as arguments, and joins together those bins which are below the threshold.
 The function should return a set of bin edges and associated probabilities, where each of the bins now has a probability that exceeds the given threshold.
 This is important to satisfy the requirement for the chi-squared test regarding the minimum number of samples per bin at the tails of the distribution of the dice sums.

    1. For example, the input bins could be:

        | bin_edge   |   probability |
        |:-----------|--------------:|
        | (1, 1)     |         0.025 |
        | (2, 2)     |         0.075 |
        | (3, 3)     |         0.2   |
        | (4, 4)     |         0.2   |
        | (5, 5)     |         0.025 |
        | (6, 6)     |         0.015 |
        | (7, 7)     |         0.2   |
        | (8, 8)     |         0.2   |
        | (9, 9)     |         0.025 |
        | (10, 10)   |         0.015 |
        | (11, 11)   |         0.015 |
        | (12, 12)   |         0.015 |

    2. For which the output bins (with a minimum probability of 0.06) would then be:

        | bin_edge   |   probability |
        |:-----------|--------------:|
        | (1, 2)     |         0.1   |
        | (3, 3)     |         0.2   |
        | (4, 5)     |         0.225 |
        | (6, 7)     |         0.215 |
        | (8, 8)     |         0.2   |
        | (9, 12)    |         0.07  |

    3. Note that some bins were merged, and their original probabilities summed: e.g. $\{(9, 9), \dots, (12, 12)\} \rightarrow (9, 12)$.

 7. Using sums of 6-sided dice, this task investigates the CLT.
 For $2, \dots, 6$ dice, compute the expected distribution of their sum, sample from this distribution randomly, and use both the bin-centre and continuity correction method from the previous tasks in order to compute probabilities for each bin (modified using the function from 6.).
 Using the expected (analytical) probabilities for each bin, compute the p-value of the chi-square test for both the bin-centre and continuity-corrected normal distribution values for each number of dice.
 Using your results, you should be able to answer the following:
    - Does the bin-centre or continuity-corrected method work better?
    - How many 6-sided dice are needed in order to confidently say that their sum is described by a normal distribution? How does this depend on the significance level?

from func import *
import mc_integration as mci
import numpy as np
import matplotlib.pyplot as plt

# number of samples
n_samples = [10, 100, 1000, 2000, 5000, 10000, 20000, 50000, 100000]

# define Monte Carlo estimators
estimators = [[mci.PlainMC(u1),
               mci.StratifiedMC(u1),
               mci.LHSMC(u1)],

              [mci.PlainMC(u2),
               mci.StratifiedMC(u2),
               mci.LHSMC(u2),
               mci.ImportanceSamplingMC(u2),
               mci.ControlVariateMC(u2, u2_control_variate, u2_cv_expectation)]]

variances = []

# Monte Carlo integration
# loop through parametric responses
for i in range(len(estimators)):
    variances.append([])
    print("estimators for u%d:" % (i+1))
    # loop through estimators for a specific parametric response
    for j in range(len(estimators[i])):
        estimator = estimators[i][j]
        variances[i].append([])
        for n in n_samples:
            # Monte Carlo integration
            estimator.run(n) 
            # print out estimator statistics
            estimator.print_results()
            # generate a plot of samples when n = 100
            if (n == 100): estimator.plot_samples()
            # record variance of the estimator
            variances[i][j].append(estimator.variance)
    print("")

# plot variance vs. n for all the estimators
for i in range(len(estimators)):
    plt.clf()
    for j in range(len(estimators[i])):
        plt.plot(n_samples,
                 variances[i][j],
                 label=estimators[i][j].name)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='lower left')
    plt.savefig('u%d_estimators_variances.png' % (i+1))
    print("variance vs. n plot for u%d(y) esitmators saved!" % (i+1))

# save variances
np.save('variances.npy', variances)

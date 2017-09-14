import mc_integration as mci
import numpy as np
import matplotlib.pyplot as plt

def u1(y):
    return 0.5 - (y[0]-0.5)**2 - (y[1]-0.5)**2

def u2(y):
    delta = 0.1
    return 1/(abs(0.3-y[0]**2-y[1]**2) + delta)

# a control variate for u2(y) 
def u2_control_variate(y):
    return 8 * np.exp(-np.abs(y[0]+y[1]-0.7)*6) + 2

# exact solution for the control variate
u2_cv_expectation = 2*(5+5*np.exp(18/5)-10*np.exp(6)+87*np.exp(39/5))/(45*np.exp(39/5))

n_samples = [100, 1000, 2000, 5000, 10000]

estimators = [[mci.PlainMC(u1),
               mci.StratifiedMC(u1),
               mci.LHSMC(u1)],

              [mci.PlainMC(u2),
               mci.StratifiedMC(u2),
               mci.LHSMC(u2),
               mci.ImportanceSamplingMC(u2),
               mci.ControlVariateMC(u2, u2_control_variate, u2_cv_expectation)]]

variances = []
for i in range(len(estimators)):
    variances.append([])
    print("estimators for u%d:" % (i+1))
    for j in range(len(estimators[i])):
        estimator = estimators[i][j]
        variances[i].append([])
        for n in n_samples:
            # Monte Carlo integration
            estimator.run(n)
            estimator.print_results()
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

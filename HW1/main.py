import mc_integration as mci

def u1(y):
    return 0.5 - (y[0]-0.5)**2 - (y[1]-0.5)**2

def u2(y):
    delta = 0.1
    return 1/(abs(0.3-y[0]**2-y[1]**2) + delta)

n_samples = [100, 1000, 2000, 5000, 10000]

estimators = [[mci.PlainMC(u1),
               mci.StratifiedMC(u1),
               mci.LHSMC(u1)],
              [mci.PlainMC(u2),
               mci.StratifiedMC(u2),
               mci.LHSMC(u2),
               mci.ImportanceSamplingMC(u2)]]

for i in range(len(estimators)):
    print("estimators for u%d:" % (i+1))
    for estimator in estimators[i]:
        for n in n_samples:
            estimator.run(n)
            estimator.print_results()
    print("")

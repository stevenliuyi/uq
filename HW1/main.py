import mc_integration as mci

def u1(y):
    return 0.5 - (y[0]-1/2)**2-(y[1]-1/2)**2

def u2(y):
    delta = 0.1
    return 1/(abs(0.3-y[0]**2-y[1]**2) + delta)

n_samples = [10, 100, 1000, 2000, 5000, 10000]

estimator = mci.PlainMC(u1)

for n in n_samples:
    estimator.run(n)
    estimator.print_results()

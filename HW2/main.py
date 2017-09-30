import numpy as np
from quasi_mc import QuasiMC

# quasi-Monte Carlo estimators
estimators = [QuasiMC('faure'),
              QuasiMC('sobol')]

# number of samples
n_samples = [10, 100, 1000, 10000]

for estimator in estimators:
    for n in n_samples:
        estimator.run(n)
        estimator.print_results()

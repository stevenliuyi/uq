import numpy as np
from quasi_mc import QuasiMC

estimator = QuasiMC('faure')
n_samples = [10, 100, 1000, 10000]

for n in n_samples:
    estimator.run(n)
    estimator.print_results()

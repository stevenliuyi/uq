import numpy as np
import seq
from h2Model import trussModel
import matplotlib.pyplot as plt

# Quasi Monte Carlo estimator for truss model
class QuasiMC:
    def __init__(self, sequence):
        self.seq_name = sequence
        self.tm = trussModel() # truss model
    
    # print out statistics
    def print_results(self):
        print("=========================================")
        print(self.display_name)
        print("number of samples: %d" % self.n_samples)
        print("mean displacement: %.8e" % self.mean)
        print("variance of displacement: %.8e" % self.variance)

    # compute mean and variance in a single-pass
    def calc_statistics(self, values):
        num = len(values)
        mu = values[0]
        var = 0.0
        for i in range(num-1):
            delta = values[i+1] - mu
            mu += 1/float(i+2) * delta
            var += (i+1)/float(i+2) * delta**2
        var /= num - 1

        # compute with numpy
        # mu = np.mean(values)
        # var = np.var(values, ddof=1)

        return (mu, var)

    # obtain sequence
    def get_sequence(self, n_samples):
        if (self.seq_name == 'halton'):
            self.display_name = 'Quasi Monte Carlo with Halton sequence'
            return seq.HaltonSeq(n_samples, self.tm.totTruss*2)
        elif (self.seq_name == 'faure'):
            self.display_name = 'Quasi Monte Carlo with Faure sequence'
            return seq.HaltonSeq(n_samples, self.tm.totTruss*2, faure=True)

    # map uniformly distributed samples in [0,1] to [a,b]
    @staticmethod
    def get_samples(samples, a, b):
        return samples * (b-a) + a

    # run quasi Monte Carlo estimator
    def run(self, n_samples):
        self.n_samples = n_samples

        # obtain sequence
        self.seq = self.get_sequence(n_samples)

        # obtain properties
        young = self.get_samples(self.seq.get()[:self.tm.totTruss,:], 190.0e9, 210.0e9)
        area = self.get_samples(self.seq.get()[self.tm.totTruss:,:], 12.0e-4, 16.0e-4)

        # solve model
        disp = list(map(lambda i: self.tm.solve(young[:,i], area[:,i])[1,2],
            range(n_samples)))

        # obtain statistics
        mean, var = self.calc_statistics(disp)
        self.mean = mean
        self.variance = var

        return

import numpy as np

class MCIntegration:
    def __init__(self, func, samples_generator):
        self.func = func
        self.samples_generator = samples_generator
    
    def print_results(self):
        print("===========================")
        print("number of samples: %d" % self.n_samples)
        print("mean of estimator: %.8f" % self.mean)
        print("variance of estimator: %.8f" % self.variance)

    def onetime_run(self, n_samples):
        # generate samples
        samples = self.samples_generator(n_samples)

        # function evaluatoin
        vals = []
        for i in range(n_samples):
            vals.append(self.func(samples[i]))

        return np.mean(vals)

    def run(self, n_samples, n_repetitions=300):
        self.n_samples = n_samples

        means = []
        for i in range(n_repetitions):
            mean = self.onetime_run(n_samples)
            means.append(mean)

        self.mean = np.mean(means)
        self.variance = np.var(means)

        return (self.mean, self.variance)

# Plain Monte Carlo estimator
class PlainMC(MCIntegration):
    def __init__(self, func):
        MCIntegration.__init__(self, func, PlainMC.samples_generator)

    @staticmethod
    def samples_generator(n_samples):
        samples = []
        for i in range(n_samples):
            sample = np.zeros(2)
            for d in range(len(sample)):
                sample[d] = np.random.uniform(0, 1)
            samples.append(sample)
        return samples

import numpy as np

class MCIntegration:
    def __init__(self, func, samples_generator, name):
        self.func = func
        self.samples_generator = samples_generator
        self.name = name
    
    def print_results(self):
        print("=====================================")
        print(self.name)
        print("number of samples: %d" % self.n_samples)
        print("mean of estimator: %.8e" % self.mean)
        print("variance of estimator: %.8e" % self.variance)

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
        name = 'Plain Monte Carlo estimator'
        MCIntegration.__init__(self, func, self.samples_generator, name)

    @staticmethod
    def samples_generator(n_samples):
        samples = []
        for i in range(n_samples):
            sample = np.zeros(2)
            for dim in range(len(sample)):
                sample[dim] = np.random.uniform(0, 1)
            samples.append(sample)
        return samples

# Stratified Monte Carlo estimator
class StratifiedMC(MCIntegration):
    def __init__(self, func, n_strata=100):
        # number of strata
        self.n_strata = n_strata
        name = 'Stratified Monte Carlo estimator'
        MCIntegration.__init__(self, func, self.samples_generator, name)

    def samples_generator(self, n_samples):
        # probability associated with each stratum for uniform grid
        prob = 1.0/self.n_strata

        # number of samples per starta
        m = int(prob * n_samples)

        # partition
        partition = np.linspace(0, 1, num=int(np.sqrt(self.n_strata))+1)

        samples = []

        for ix in range(len(partition)-1):
            for iy in range(len(partition)-1):
                x_left = partition[ix]
                x_right = partition[ix+1]

                y_left = partition[iy]
                y_right = partition[iy+1]

                for i in range(m):
                    sample = np.zeros(2)
                    sample[0] = np.random.uniform(x_left, x_right)
                    sample[1] = np.random.uniform(y_left, y_right)
                    samples.append(sample)
        
        return samples

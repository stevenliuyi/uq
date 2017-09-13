import numpy as np

class MCIntegration:
    def __init__(self, func, samples_generator, name, preconditioner=None):
        self.func = func
        self.samples_generator = samples_generator
        self.name = name
        self.preconditioner = preconditioner
    
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

        if self.preconditioner is not None:
            vals /= self.preconditioner

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

        # number of samples per strata
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

# Latin Hypercube Monte Carlo estimator
class LHSMC(MCIntegration):
    def __init__(self, func):
        name = 'Latin Hypercube Monte Carlo estimator'
        MCIntegration.__init__(self, func, self.samples_generator, name)

    @staticmethod
    def samples_generator(n_samples):

        samples = []
        dims = 2
        # uniform random permutation of {0,1,...,M-1}
        pi = list(map(lambda x: np.random.permutation(n_samples), range(dims)))

        for i in range(n_samples):
            sample = np.zeros(dims)
            for dim in range(dims):
                u = np.random.uniform(0, 1)
                sample[dim] = (pi[dim][i] + u) / n_samples
            samples.append(sample)

        return samples

# Importance Sampling Monte Carlo estimator
class ImportanceSamplingMC(MCIntegration):
    def __init__(self, func):
        name = 'Importance Sampling Monte Carlo estimator'
        MCIntegration.__init__(self, func, self.samples_generator, name)

    # use trapezoidal rule to approximate optimal pdf
    def modified_pdf_generator(self, n_partitions=100):
        # number of partitions in each direction
        p = int(np.sqrt(n_partitions))

        partition = np.linspace(0, 1, num=p+1)
        
        pdf = np.zeros((p, p))
        e = 0.0 # approximation of E[g(y)]
        for ix in range(p):
            for iy in range(p):
                x_left = partition[ix]
                x_right = partition[ix+1]

                y_left = partition[iy]
                y_right = partition[iy+1]

                x = np.random.uniform(x_left, x_right)
                y = np.random.uniform(y_left, y_right)
                pdf[ix, iy] = self.func([x, y])
                e += pdf[ix, iy] * 1.0 / n_partitions
        pdf /= e

        return pdf
        
    def samples_generator(self, n_samples):
        # obtain modified pdf
        modified_pdf = self.modified_pdf_generator()

        # number of partitions in each direction
        p = modified_pdf.shape[0]
        partition = np.linspace(0, 1, num=p+1)

        # pdf in x direction
        pdf_x = np.sum(modified_pdf, axis=1) / p
        
        # cpf in x direction
        cdf_x = np.array(list(map(lambda x: np.sum(pdf_x[0:(x+1)]), range(p)))) / p

        samples = []
        preconditioner = [] # preconditioner
        for i in range(n_samples):
            # use inverse cdf to sample x
            ux = np.random.uniform(0,1)
            ix = np.searchsorted(cdf_x, ux)
            x = np.random.uniform(partition[ix], partition[ix+1])

            # pdf in y direction
            pdf_y  = modified_pdf[ix,:]

            # cdf in y direction
            cdf_y = np.array(list(map(lambda x: np.sum(pdf_y[0:(x+1)]), range(p)))) / np.sum(pdf_y)

            # use inverse cdf to sample y
            uy = np.random.uniform(0,1)
            iy = np.searchsorted(cdf_y, uy)
            y = np.random.uniform(partition[iy], partition[iy+1])

            sample = np.array([x, y])
            samples.append(sample)

            # set preconditioner
            preconditioner.append(modified_pdf[ix,iy])

        self.preconditioner = np.array(preconditioner)
        return samples

import numpy as np
import matplotlib.pyplot as plt

class MCIntegration:
    def __init__(self, func, samples_generator, name):
        self.func = func
        self.samples_generator = samples_generator
        self.name = name
    
    # print out statistics
    def print_results(self):
        print("=========================================")
        print(self.name)
        print("number of samples: %d" % self.n_samples)
        print("mean of estimator: %.8e" % self.mean)
        print("variance of estimator: %.8e" % self.variance)

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

    # run Monte Carlo integration once
    def onetime_run(self, n_samples):
        # generate samples
        self.samples = self.samples_generator(n_samples)

        # function evaluatoin
        vals = list(map(lambda i: self.func(self.samples[i]), range(n_samples)))

        # only for importance sampling
        if hasattr(self, 'preconditioner'):
            vals /= self.preconditioner

        # only for control variate
        if hasattr(self, 'beta'):
            vals += self.beta * self.cv_expectation

        # obtain statistics
        mean, var = self.calc_statistics(vals)

        return mean

    # run Monte Carlo integration for many times and obtain statistics
    def run(self, n_samples, n_repetitions=300):
        self.n_samples = n_samples

        means = list(map(lambda i: self.onetime_run(n_samples), range(n_repetitions)))

        # mean and variance of the estimators
        self.mean, self.variance = self.calc_statistics(means) 

        return

    # plot generated samples
    def plot_samples(self):
        samples_coord = np.array(self.samples).transpose()
        plt.clf()
        plt.plot(samples_coord[0], samples_coord[1], 'bo')
        plt.axis([0,1,0,1])
        plt.title('%s - %d samples' % (self.name, self.n_samples))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xticks(np.linspace(0, 1, 11))
        plt.yticks(np.linspace(0, 1, 11))
        plt.grid(b=True, color='k', linestyle='--')
        plt.savefig('%s - %d samples' % (self.name, self.n_samples))
        print("=========================================")
        print("samples plot saved!")


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

    @staticmethod
    def single_stratum_samples_generator(samples, partition, ix, iy, n):
        x_left = partition[ix]
        x_right = partition[ix+1]

        y_left = partition[iy]
        y_right = partition[iy+1]

        for i in range(n):
            sample = np.zeros(2)
            sample[0] = np.random.uniform(x_left, x_right)
            sample[1] = np.random.uniform(y_left, y_right)
            samples.append(sample)

        return samples

    def samples_generator(self, n_samples):
        # probability associated with each stratum for uniform grid
        prob = 1.0/self.n_strata

        # number of samples per strata
        m = int(prob * n_samples)

        # partition
        n_strata_one_direction = int(np.sqrt(self.n_strata))
        partition = np.linspace(0, 1, num=n_strata_one_direction+1)

        samples = []

        for ix in range(len(partition)-1):
            for iy in range(len(partition)-1):
                samples = self.single_stratum_samples_generator(samples,
                                                                partition,
                                                                ix, iy, m)

        # remain number of samples
        n_samples_remain = n_samples - self.n_strata**2 * m

        # non-zero when the number of samples is not multiple of the number of strata
        if n_samples_remain > 0:
            remains = np.random.choice(self.n_strata, n_samples_remain, replace=False)
            for r in remains:
                ix = int(r/n_strata_one_direction) 
                iy = r%n_strata_one_direction

                samples = self.single_stratum_samples_generator(samples,
                                                                partition,
                                                                ix, iy, 1)
        
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

# Control Variate Monte Carlo estimator
class ControlVariateMC(MCIntegration):
    def __init__(self, func, control_variate, cv_expectation):
        self.original_func = func
        self.control_variate = control_variate
        self.cv_expectation = cv_expectation
        name = 'Control Variate Monte Carlo estimator'

        MCIntegration.__init__(self, self.new_func, self.samples_generator, name)

    def new_func(self, y):
            return self.original_func(y) - self.beta * self.control_variate(y)

    # estimator optimal coefficient
    def calc_optimal_coefficient(self, n_samples):
        g = []
        h = []
        for i in range(n_samples):
            y = np.zeros(2)
            for dim in range(len(y)):
                y[dim] = np.random.uniform(0, 1)
            g.append(self.original_func(y))
            h.append(self.control_variate(y))
        
        cov = np.cov(g, h)
        beta = cov[0,1] / cov[1,1]
        return beta
        
    def samples_generator(self, n_samples):
        self.beta = self.calc_optimal_coefficient(n_samples)

        samples = []
        for i in range(n_samples):
            sample = np.zeros(2)
            for dim in range(len(sample)):
                sample[dim] = np.random.uniform(0, 1)
            samples.append(sample)
        return samples

import numpy as np
import sklearn.gaussian_process as gp
from scipy import stats
import matplotlib.pyplot as plt
from optimization import *

class BayesianOptimization(Optimization):
    def __init__(self, bounds, kernel='m52', opt_method=None):

        # bounds for all parameters
        self.bounds = np.array(bounds)

        # number of parameters
        self.dim = self.bounds.shape[0]
        
        self.x_next = None

        # kernel for Guassian Process
        if kernel == 'm52':
            # Matern 5/2 kernel
            kernel = gp.kernels.Matern(nu=2.5)
        elif kernel == 'se':
            # squared exponential kernel
            kernel = gp.kernels.RBF()
            
        # Gaussian Process regressor
        self.gpr = gp.GaussianProcessRegressor(kernel=kernel,
                                               n_restarts_optimizer=10,
                                               normalize_y=True)

        # optimization method
        super().__init__(opt_method)

    # initial points
    def initialize(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)

        self.minimize()

    # udpate y value for x_next
    def update(self, y, x=None):
        if x is not None: self.x_next = x

        self.x = np.vstack((self.x, self.x_next.reshape((1,-1))))
        self.y = np.append(self.y, y)

        self.minimize()

    def minimize(self):
        # update the best y
        self.ybest = np.amin(self.y)
        self.xbest = self.x[np.argmin(self.y)]

        # fit Gaussian Process
        self.gpr.fit(self.x, self.y)

        self.x_next, _ = self.optimization(lambda x: -self.acq_func(x.reshape((1,-1)), self.ybest),
                                           self.bounds)

    # acquisition function (expected improvement)
    def acq_func(self, x, ymin):
        mean, std = self.gpr.predict(x, return_std=True)
        if std == 0: std = 1e-10 # prevent NaN for gamma
        gamma = (ymin - mean) / std
        return std*(gamma*stats.norm.cdf(gamma) + stats.norm.pdf(gamma))

    def plot(self):
        if self.dim == 1:
            x = np.linspace(self.bounds[0,0],self.bounds[0,1], num=100)
            mean, std = self.gpr.predict(x.reshape(-1,1), return_std=True)
            plt.figure()
            plt.plot(x, mean)
            plt.fill_between(x, mean+std, mean-std, alpha=0.1)
            plt.scatter(self.x.flatten(), self.y)
            plt.xlim(x[0], x[-1])
            plt.show()
        else:
            print('only 1D data are supported for plotting!')

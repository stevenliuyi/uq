import numpy as np
import scipy as sp
from optimization import *

class RBFOptimization(Optimization):
    def __init__(self, x, y, kernel='inverse', opt_method=None):
        self.x = np.array(x)
        self.y = np.array(y)
        self.n = x.shape[0]
        self.dim = x.shape[1]

        self.kernel = kernel

        # kernel matrices
        m = self.get_kernel_matrices()

        # calculate coefficients
        self.alpha = np.linalg.solve(m, y).flatten()

        # optimization method
        super().__init__(opt_method)

    def kernel_func(self, x, y):
        if self.kernel == 'inverse': # inverse-quadratic RBF
            return np.sqrt(1.0+np.linalg.norm(x-y,ord=2)**2)
        elif self.kernel == 'gaussian': # Guassian RBF
            return np.exp(-np.linalg.norm(x-y,ord=2)**2)

    def get_kernel_matrices(self):
        m = np.zeros((self.n, self.n))
        for i in range(self.n):
            for k in range(self.n):
                m[i,k] = self.kernel_func(self.x[i,:],self.x[k,:])
        return m

    def interpolate(self, xi):
        yi = 0
        for k in range(self.n):
            yi += self.alpha[k] * self.kernel_func(xi,self.x[k,:])

        return yi

    def minimize(self, bounds):
        bounds = np.array(bounds)

        self.xbest, self.ybest = self.optimization(self.interpolate, bounds)

        return self.ybest

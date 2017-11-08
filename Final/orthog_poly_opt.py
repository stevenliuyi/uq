import numpy as np
import chaospy as cp

class OrthogPolyOptimization:
    def __init__(self, x, y, rho='uniform', order=None, support=None, a=0.5, b=0.5, mu=0, sigma=1, shape=1, scale=1):
        self.x = np.array(x)
        self.y = np.array(y)

        self.dim = self.x.shape[1]

        support = self.get_support(rho, support)

        # order of polynomials
        self.order = order if order is not None else self.x.shape[0]

        # i.i.d. rho(x)
        dist1d = []
        for d in range(self.dim):
            if rho == 'uniform': # Legendre chaos
                dist1d.append(cp.Uniform(support[d,0], support[d,1]))
            elif rho == 'beta': # Jacobi chaos
                dist1d.append(cp.Beta(a,b,support[d,0], support[d,1]))
            elif rho == 'gaussian': # Hermite chaos
                dist1d.append(cp.Normal(mu=mu, sigma=sigma))
            elif rho == 'gamma': # Leguerre chaos
                dist1d.append(cp.Gamma(shape=shape, scale=scale))

        # joint distribution
        dist = cp.J(*dist1d)

        # orthogonal polynomials
        orthog_polys = cp.orth_ttr(self.order, dist)

        # polynomial chaos expansion
        self.yfit = cp.fit_regression(orthog_polys, self.x.T, self.y)

    @staticmethod
    def get_support(rho, support):
        if rho == 'uniform': # Legendre chaos
            if support is None:
                support = np.vstack((-np.ones(self.dim), np.ones(self.dim)))
            else:
                support = np.array(support)
        elif rho == 'beta': # Jacobi chaos
            if support is None:
                support = np.vstack((np.zeros(self.dim), np.ones(self.dim)))
            else:
                support = np.array(support)

        return support

    def predict(self, xi):
        return self.yfit(xi)

    def minimize(self, bounds):
        bounds = np.array(bounds)

        # generate Monte Carlo points
        x_mc = np.random.uniform(bounds[:,0],
                                 bounds[:,1],
                                 size=(10000,self.dim))
        y_mc = [self.predict(x) for x in x_mc]  

        self.xbest = x_mc[np.argmin(y_mc)]
        self.ybest = np.amin(y_mc)

        return self.ybest

# obtain orthogonal polynomials using three-term recurrence
import numpy as np
from scipy import integrate

class OrthogPoly:
    def __init__(self, n, a, b, weight_func, quad_rule=None):
        self.n = n # order of polynomial
        self.a = a # lower bound
        self.b = b # upper bound
        self.weight_func = weight_func

        # nodes and weights for integration
        self.quad_rule = quad_rule

        self.alphas = []
        self.betas = []

        self.generate_polys()
        
    @staticmethod
    def poly_func(f):
        return lambda z: np.polyval(f, z)

    def inner_product(self, f1, f2):
        if (self.quad_rule is None):
            # quad rule not provided
            # use scipy.integrate.quad to calculate integration
            return integrate.quad(lambda y: f1(y)*f2(y)*self.weight_func(y),
                                  self.a, self.b)[0]
        else:
            # use provided nodes and weights to calculate integration
            nodes = self.quad_rule[0]
            weights = self.quad_rule[1]
            return np.sum(weights * [f1(y)*f2(y)*self.weight_func(y) \
                    for y in nodes])
            

    def length(self, f):
        return np.sqrt(self.inner_product(f, f))

    def generate_polys(self):
        self.polys = [np.poly1d([0]) for i in range(self.n+1)]

        self.polys[0] = np.poly1d([1]) # p_0(y)
        
        self.alphas = []; self.betas = []
        for n in range(1, self.n+1):
            pn1 = self.polys[n-1] # p_(n-1)(y)
            pn2 = self.polys[n-2] if n != 1 else np.poly1d([0]) # p_(n-2)(y)
            
            ypn1 = pn1 * np.poly1d([1,0])
            # <p_(n-1), y*p_(n-1)>
            inner1 = self.inner_product(self.poly_func(pn1),
                                        self.poly_func(ypn1))
            # <p_(n-1), p_(n-1)>
            inner2 = self.length(self.poly_func(pn1))**2
            # <p_(n-2), p_(n-2)>
            inner3 = self.length(self.poly_func(pn2))**2
            # alpha_(n-1)
            alpha = inner1 / inner2
            # beta_(n-1)
            beta = inner2 / inner3 if n != 1 else 0.0

            # obtain p_n(y) by three-term recurrence
            pn = np.poly1d([1,-alpha])*pn1 - beta*pn2
            self.polys[n] = pn

            # record alpha and beta
            self.alphas.append(alpha)
            self.betas.append(beta)
        
        # normalization
        for n in range(0,self.n+1):
            pn = self.polys[n]
            self.polys[n] = pn / self.length(self.poly_func(pn))

        # set beta0
        self.betas[0] = 1 / self.polys[0].c[0]**2

    # return the Jacobi matrix
    def get_jacobi(self):
        jacobi = np.zeros((self.n, self.n))
        for i in range(0, self.n):
            jacobi[i, i] = self.alphas[i]

        for i in range(1, self.n):
            jacobi[i-1,i] = np.sqrt(self.betas[i])
            jacobi[i,i-1] = np.sqrt(self.betas[i])

        return jacobi

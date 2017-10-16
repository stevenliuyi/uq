import numpy as np
import matplotlib.pyplot as plt
from clenshaw_curtis import *
from orthog_poly import *
from scipy.special import gamma

order = 9 # obtain orthogonal polynomials up to this order
bounds = [1, 5] # lower and upper bounds
alpha = 2; beta = 5 # parameters for beta distribution

# Clenshaw-Curtis rule
cc_order = order*2+1+(alpha-1)+(beta-1)
cc = clenshaw_curtis(bounds[0], bounds[1], cc_order+1)
print('order of Clenshaw-Curtis quadrature rule: %d' % cc_order)

# weight function (beta distribution)
loc = bounds[0]
scale = bounds[1] - bounds[0]
weight_func = lambda y: gamma(alpha+beta)/gamma(alpha)/gamma(beta) \
        *((y-loc)/scale)**(alpha-1) * (1-(y-loc)/scale)**(beta-1)

# use Clenshaw-Curtis to obtain orthogonal polynomials
polys = OrthogPoly(order, bounds[0], bounds[1], weight_func, cc).polys

# for comparison, use scipy.integrate.quad to obtain orthogonal polynomials
sp_polys = OrthogPoly(order, bounds[0], bounds[1], weight_func).polys

print('coefficent distance norm: %e' % (np.linalg.norm(polys[order].c-sp_polys[order].c,2)))

x = np.linspace(bounds[0], bounds[1], 100)
print('')
print('coefficients:')
for poly in polys:
    print('%dth order polynomial: %s' % (poly.order, ', '.join(map(str, poly.c))))
    plt.plot(x, np.polyval(poly, x), label='order %d' % poly.order)
plt.yscale('symlog')
plt.legend(loc='upper left', ncol=3)
plt.title('first %d orthogonal polynomials' % (order+1))
plt.savefig('first_%d_orthgonal_polynomials.png' % (order+1))
print('polynomial plot is saved!')


# Golub-Welsch algorithm
orthog_poly = OrthogPoly(order, bounds[0], bounds[1], weight_func, cc)
jacobi = orthog_poly.get_jacobi()
x, v = np.linalg.eig(jacobi) # x is list of nodes (eigenvalues)

# weights
beta0 = orthog_poly.betas[0]
w = v[0,:]**2 * beta0

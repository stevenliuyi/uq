import numpy as np
import matplotlib.pyplot as plt
from quad import *
from orthog_poly import *
from scipy.special import gamma

order = 9 # obtain orthogonal polynomials up to this order
bounds = [1, 5] # lower and upper bounds
alpha = 2; beta = 5 # parameters for beta distribution

# Clenshaw-Curtis rule
cc_order = order*2+(alpha-1)+(beta-1)
cc = clenshaw_curtis(bounds[0], bounds[1], cc_order+1)
print('order of Clenshaw-Curtis quadrature rule: %d' % cc_order)

# weight function (beta distribution)
loc = bounds[0]
scale = bounds[1] - bounds[0]
weight_func = lambda y: gamma(alpha+beta)/gamma(alpha)/gamma(beta) \
        *((y-loc)/scale)**(alpha-1) * (1-(y-loc)/scale)**(beta-1)

# use Clenshaw-Curtis to obtain orthogonal polynomials
polys = OrthogPoly(order, bounds[0], bounds[1], weight_func, cc)

x = np.linspace(bounds[0], bounds[1], 100)
print('')
print('coefficients:')
for poly in polys.polys:
    print('%dth order polynomial: %s' % (poly.order, ', '.join(map(str, poly.c))))
    plt.plot(x, np.polyval(poly, x), label='order %d' % poly.order)
plt.yscale('symlog')
plt.legend(loc='upper left', ncol=3)
plt.title('first %d orthonormal polynomials' % (order+1))
plt.savefig('first_%d_orthonormal_polynomials.png' % (order+1))

# Gauss quad rule
nodes, weights = gauss_with_polys(polys)
print('')
print('Gauss quad rule')
print('nodes:')
print(nodes)
print('weights:')
print(weights)
plt.clf()
plt.plot(nodes, weights)
plt.title('nodes and weights for Gauss quad rule')
plt.savefig('Gauss_quad_rule.png')
print('Gauss quad rule plot is saved!')

# Guass-Kronrod quad rule
for n in [2, 3]:
    nodes, weights = gauss_kronrod_with_polys(n, polys)
    print('')
    print('Gauss-Kronrod quad rule (%d points)' % (2*n+1))
    print('nodes:')
    print(nodes)
    print('weights:')
    print(weights)
    plt.clf()
    plt.plot(nodes, weights)
    plt.title('nodes and weights for Gauss-Kronrod quad rule (%d points)' % (2*n+1))
    plt.savefig('Gauss_Kronrod_quad_rule_%d_points.png' % (2*n+1))
    print('Gauss-Kronrod quad rule (%d points) plot is saved!' % (2*n+1))

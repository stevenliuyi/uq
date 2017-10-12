import numpy as np
import matplotlib.pyplot as plt
from clenshaw_curtis import *
from orthog_poly import *
from scipy.special import legendre, jacobi, chebyt

# -----------------------------
# test Clenshaw Curtis rule
print('===========================')
print('Clenshaw-Curtis rule test')
print('===========================')
print('')

points, weights = clenshaw_curtis(-1,1,9)

ex_points = [1.0000000000000000,
             0.9238795325112867,
             0.7071067811865475,
             0.3826834323650897,
             0.0000000000000000,
            -0.3826834323650898,
            -0.7071067811865475,
            -0.9238795325112867,
            -1.0000000000000000]

ex_weights = [0.1587301587301588E-01,
              0.1462186492160182,
              0.2793650793650794,
              0.3617178587204897,
              0.3936507936507936,
              0.3617178587204898,
              0.2793650793650794,
              0.1462186492160182,
              0.1587301587301588E-01]

print('point distance norm: %e' % (np.linalg.norm(points-ex_points, 2)))
print('weight distance norm: %e' % (np.linalg.norm(weights-ex_weights, 2)))

# plot nodes and weights for comparison
plt.plot(points, weights, label='computed')
plt.plot(ex_points, ex_weights, label='benchmark')
plt.title('nodes and weights for the Clenshaw Curtis rule')
plt.legend(loc='lower left')
plt.show()

# -----------------------------
# test orthogonal polynomials
print('')
print('===========================')
print('Orthogonal polynomials test')
print('===========================')
print('')

def test_orthog_poly(order, a, b, weight_func, quad_rule, exact_poly, name):
    # generate orthogonal polynomials
    orthogPoly1 = OrthogPoly(order, a, b, weight_func)
    if (quad_rule is not None):
        orthogPoly2 = OrthogPoly(order, a, b, weight_func, quad_rule)

    print('===== %s =====' % name)

    for i in range(0, len(orthogPoly1.polys)):
        monic_poly1 = orthogPoly1.polys[i] / orthogPoly1.polys[i].c[0]
        if (quad_rule is not None):
            monic_poly2 = orthogPoly2.polys[i] / orthogPoly2.polys[i].c[0]

        print("===== %dth order =====" % i)
        print("coefficients: %s" % ', '.join(map(str, monic_poly1.c)))
        if (quad_rule is not None):
            print("coefficient distance norm 1: %e" % \
                    np.linalg.norm(monic_poly1.c-monic_poly2.c, 2))
            print("coefficient distance norm 2: %e" % 
                    np.linalg.norm(monic_poly2.c-exact_poly(i).c, 2))
        else:
            print("coefficient distance norm: %e" % \
                    np.linalg.norm(monic_poly1.c-exact_poly(i).c, 2))


    # plot
    x = np.linspace(a, b, 100)
    plt.clf()
    plt.plot(x, np.polyval(monic_poly1, x), label='numerical')
    if (quad_rule is not None):
        plt.plot(x, np.polyval(monic_poly2, x), label='numerical (Clenshaw-Curtis)')
    plt.plot(x, np.polyval(exact_poly(order), x), label='exact')
    plt.title(name)
    plt.legend(loc='lower left')
    plt.show()
    print('')

# Legendre polynomials
test_orthog_poly(5, -1, 1,
                 lambda x: 1,
                 # highest order in integrand is 4+5=9
                 clenshaw_curtis(-1,1,10),
                 lambda n: legendre(n,monic=True),
                 'Legendre polynomials')

# Jacobi polynomials
alpha = 1.0; beta = 2.0
test_orthog_poly(5, -1, 1,
                 lambda x: (1.0-x)**alpha * (1.0+x)**beta,
                 # highest order in integrand is 4+5+3=12
                 clenshaw_curtis(-1,1,13), 
                 lambda n: jacobi(n,alpha,beta,monic=True),
                 'Jacobi polynomials')

# Chebyshev polynomials
test_orthog_poly(5, -1, 1,
                 lambda x: (1-x**2)**(-0.5),
                 None,
                 lambda n: chebyt(n, monic=True),
                 'Chebyshev polynomials')

import numpy as np
import matplotlib.pyplot as plt
from quad import *
from orthog_poly import *
from scipy.special import legendre, jacobi, chebyt, gamma

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

def test_orthog_poly(order, a, b, weight_func, weight_func_poly, quad_rule, exact_poly, name):
    ## generate orthogonal polynomials

    # integrate using scipy.integrate.quad
    sp_orthogPoly = OrthogPoly(order, a, b, weight_func) 
    # integrate using np.polyint (integration would be exact)
    if (weight_func_poly is not None):
        ex_orthogPoly = OrthogPoly(order, a, b, weight_func_poly)
    # integrate using provided quad rule
    if (quad_rule is not None):
        cc_orthogPoly = OrthogPoly(order, a, b, weight_func, quad_rule)

    print('===== %s =====' % name)

    for i in range(0, len(sp_orthogPoly.polys)):
        sp_monic_poly = sp_orthogPoly.polys[i] / sp_orthogPoly.polys[i].c[0]
        if (weight_func_poly is not None):
            ex_monic_poly = ex_orthogPoly.polys[i] / ex_orthogPoly.polys[i].c[0]
        if (quad_rule is not None):
            cc_monic_poly = cc_orthogPoly.polys[i] / cc_orthogPoly.polys[i].c[0]

        # print out polynomial coefficients
        print("===== %dth order =====" % i)
        if (weight_func_poly is not None):
            print("coefficients: %s" % ', '.join(map(str, ex_monic_poly.c)))
        else:
            print("coefficients: %s" % ', '.join(map(str, sp_monic_poly.c)))


        # print out errors
        print("coefficient distance norm (scipy): %e" % \
            np.linalg.norm(sp_monic_poly.c-exact_poly(i).c, 2))
        if (weight_func_poly is not None):
            print("coefficient distance norm (polyint): %e" % \
                np.linalg.norm(ex_monic_poly.c-exact_poly(i).c, 2))
        if (quad_rule is not None):
            print("coefficient distance norm (quad rule): %e" % \
                np.linalg.norm(cc_monic_poly.c-exact_poly(i).c, 2))

    # plot
    x = np.linspace(a, b, 100)
    plt.clf()
    plt.plot(x, np.polyval(sp_monic_poly, x), label='numerical (scipy)')
    if (weight_func_poly is not None):
        plt.plot(x, np.polyval(ex_monic_poly, x), label='numerical (polyint)')
    if (quad_rule is not None):
        plt.plot(x, np.polyval(cc_monic_poly, x), label='numerical (Clenshaw-Curtis)')
    plt.plot(x, np.polyval(exact_poly(order), x), label='exact')
    plt.title(name)
    plt.legend(loc='lower left')
    plt.show()
    print('')

    if (weight_func_poly is not None): return ex_orthogPoly

# Legendre polynomials
test_orthog_poly(5, -1, 1,
                 lambda x: 1,
                 np.poly1d([1]),
                 # highest order in integrand is 4+5=9
                 clenshaw_curtis(-1,1,10),
                 lambda n: legendre(n,monic=True),
                 'Legendre polynomials')

# Jacobi polynomials
alpha = 1.0; beta = 2.0
test_orthog_poly(5, -1, 1,
                 lambda x: (1.0-x)**alpha * (1.0+x)**beta,
                 np.poly1d([-1,1])**int(alpha) * np.poly1d([1,1])**int(beta),
                 # highest order in integrand is 4+5+3=12
                 clenshaw_curtis(-1,1,13), 
                 lambda n: jacobi(n,alpha,beta,monic=True),
                 'Jacobi polynomials')

# Chebyshev polynomials
test_orthog_poly(5, -1, 1,
                 lambda x: (1-x**2)**(-0.5),
                 None,
                 None,
                 lambda n: chebyt(n, monic=True),
                 'Chebyshev polynomials')

# orthogonal polynomials wrt beta distribution
alpha = 2; beta = 5; loc = 1; scale = 4
weight_func = np.poly1d([1/scale,-loc/scale])**int(alpha-1) \
        * np.poly1d([-1/scale,1+loc/scale])**int(beta-1) \
        * gamma(alpha+beta)/gamma(alpha)/gamma(beta)
orthogPoly = OrthogPoly(9, 1, 5, weight_func)

x = np.linspace(1,5,100)
for poly in orthogPoly.polys:
    plt.plot(x, np.polyval(poly, x), label='order %d' % poly.order)
plt.yscale('symlog')
plt.legend(loc='upper left', ncol=3)
plt.title('first 10 orthogonal polynomials wrt beta distribution')
plt.show()
print('')
print('===== Orthogonal polynomials wrt beta distribution =====')
print('Check orthogonality:')
for i in range(10):
    print('<p%d, p9> = %f' % (i, \
            orthogPoly.inner_product(orthogPoly.polys[i], \
            orthogPoly.polys[9])))

# -----------------------------
# test Gauss quad rule
print('')
print('===========================')
print('Gauss quad rule test')
print('===========================')
print('')
points, weights = gauss(-1,1,4)

ex_points = [-np.sqrt((15+2*np.sqrt(30))/35),
             -np.sqrt((15-2*np.sqrt(30))/35),
              np.sqrt((15-2*np.sqrt(30))/35),
              np.sqrt((15+2*np.sqrt(30))/35)]

ex_weights = [(18-np.sqrt(30))/36,
              (18+np.sqrt(30))/36,
              (18+np.sqrt(30))/36,
              (18-np.sqrt(30))/36]

print('point distance norm: %e' % (np.linalg.norm(points-ex_points, 2)))
print('weight distance norm: %e' % (np.linalg.norm(weights-ex_weights, 2)))

# plot nodes and weights for comparison
plt.plot(points, weights, label='computed')
plt.plot(ex_points, ex_weights, label='benchmark')
plt.title('nodes and weights for the Gauss-Legendre rule')
plt.legend(loc='lower left')
plt.show()

# obtain Gauss nodes and weights wrt beta distribution
points, weights = gauss_with_polys(orthogPoly)
plt.plot(points, weights)
plt.title('nodes and weights for the Gauss quad rule')
plt.show()

print('')
print('===== use orthogonal polynomials wrt beta distribution =====')
integration = np.sum(weights*[1 for x in points])
print('integrate 1 wrt beta distribtion from 1 to 5: %f' % integration)
print('error: %e' % (integration-4))

# -----------------------------
# test Gauss-Kronrod quad rule
print('')
print('===========================')
print('Gauss-Kronrod rule test')
print('===========================')
print('')
points, weights = gauss_kronrod(-1,1,7)

ex_points = [-0.991455371120813,
             -0.949107912342759,
             -0.864864423359769,
             -0.741531185599394,
             -0.586087235467691,
             -0.405845151377397,
             -0.207784955007898,
              0.000000000000000,
              0.207784955007898,
              0.405845151377397,
              0.586087235467691,
              0.741531185599394,
              0.864864423359769,
              0.949107912342759,
              0.991455371120813]

ex_weights = [0.022935322010529,
              0.063092092629979,
              0.104790010322250,
              0.140653259715525,
              0.169004726639267,
              0.190350578064785,
              0.204432940075298,
              0.209482141084728,
              0.204432940075298,
              0.190350578064785,
              0.169004726639267,
              0.140653259715525,
              0.104790010322250,
              0.063092092629979,
              0.022935322010529]

print('point distance norm: %e' % (np.linalg.norm(points-ex_points, 2)))
print('weight distance norm: %e' % (np.linalg.norm(weights-ex_weights, 2)))

# plot nodes and weights for comparison
plt.plot(points, weights, label='computed')
plt.plot(ex_points, ex_weights, label='benchmark')
plt.title('nodes and weights for the Gauss-Kronrod rule')
plt.legend(loc='lower left')
plt.show()

# obtain Gauss-Kronrod nodes and weights wrt beta distribution
for order in [3]:
    points, weights = gauss_kronrod_with_polys(order,orthogPoly)
    plt.plot(points, weights)
    plt.title('nodes and weights for the Gauss-Kronrod quad rule (%d points)' % (2*order+1))
    plt.show()
    
    print('')
    print('===== use orthogonal polynomials wrt beta distribution =====')
    integration = np.sum(weights*[1 for x in points])
    print('integrate 1 wrt beta distribtion from 1 to 5: %f' % integration)
    print('error: %e' % (integration-4))

    # obtain corresponding Gauss nodes and weights 
    orthogPoly = OrthogPoly(order, 1, 5, weight_func=weight_func)
    points_gauss, weights_gauss = gauss_with_polys(orthogPoly)

    # make sure Gauss-Kronode nodes contain Gauss nodes
    print('')
    print('===== corresponding Gauss points =====')
    print('point distance norm: %e' % (np.linalg.norm(points[1:len(points):2]-points_gauss, 2)))



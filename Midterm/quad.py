import numpy as np
from orthog_poly import *

# Clenshaw-Curtis quad rule
def clenshaw_curtis(a, b, n):
    size = b - a

    # coefficients
    c = np.zeros([2, 2*n-2])
    c[0,0] = 2.0
    c[1,1] = 1.0
    c[1,-1] = 1.0

    idx = np.arange(2,n,2)
    c[0,idx] = 2.0/(1-idx**2)
    c[0,2*n-2-idx] = 2.0/(1-idx**2)

    # inverse Fourier Transform
    f = np.real(np.fft.ifft(c))

    # weights
    w = f[0,:n]
    w[0] *= 0.5
    w[-1] *= 0.5
    w *= size

    # nodes
    x = 0.5*((b+a) + (n-1)*size*f[1,:n])

    return x, w


# Gauss quad rule
def gauss(a, b, n, weight_func=None):

    if weight_func is None:
        weight_func = np.poly1d([1])

    orthog_polys = OrthogPoly(n, a, b, weight_func)

    return gauss_with_polys(orthog_polys)
    
# Golub-Welsch algorithm (orthogonal polynomials provided)
def gauss_with_polys(orthog_polys): 
    # Jacobi matrix
    jacobi = orthog_polys.get_jacobi()

    # obtain eigenvalues and eigenvectors of Jacobi matrix
    x, v = np.linalg.eig(jacobi) # x is list of nodes (eigenvalues)
    
    # weights
    beta0 = orthog_polys.betas[0]
    w = v[0,:]**2 * beta0

    # sort nodes and weights
    sort_index = np.argsort(x)

    return x[sort_index], w[sort_index]

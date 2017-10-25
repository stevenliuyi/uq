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

# Gauss-Kronrod quad rule
def gauss_kronrod(a, b, n, weight_func=None):
    if weight_func is None:
        weight_func = np.poly1d([1])

    # orthogonal polynomials for Gauss quad rule
    orthog_polys = OrthogPoly(int(np.ceil(1.5*n))+1, a, b, weight_func)

    return gauss_kronrod_with_polys(n, orthog_polys)

# Laurie's algorithm (orthogonal polynomials for old system provided)
def gauss_kronrod_with_polys(n, orthog_polys):

    # a and b initialization
    a = np.zeros(2*n+1)
    b = np.zeros(2*n+1)
    # first 3n+1 coefficients a0,b1,a1,b2... are given
    a_idx = int(np.floor(1.5*n))
    b_idx = int(np.ceil(1.5*n))
    a[:(a_idx+1)] = orthog_polys.alphas[:(a_idx+1)]
    b[:(b_idx+1)] = orthog_polys.betas[:(b_idx+1)]

    # initialization
    sigma = np.zeros((n+2, n+2))
    # sigma_(0,0)
    sigma[1,1] = 1.0
    # sigma_(-1,l), l = 0,...,n
    for l in range(n+1): sigma[0,l+1] = 0.0
    # sigma_(k,n), k = 1,...,n-1
    for k in range(1,n): sigma[k+1,n+1] = 0.0
    # sigma_(k,l), l=1,...,n-2, k=l+1,l+2
    for l in range(1,n-1):
        sigma[l+2,l+1] = 0.0
        sigma[l+3,l+1] = 0.0

    # eastward phase (Salzer's algorithm)
    for m in range(n-1):
        for k in range(int(np.ceil(m/2.0)),-1,-1):
            l = m - k
            sigma[k+1,l+2] = sigma[k+2,l+1] + (a[k+n+1]-a[l])*sigma[k+1,l+1] \
                    + b[k+n+1]*sigma[k,l+1] - b[l]*sigma[k+1,l]

    # southward phase (Sack-Donovan-Wheeler algorithm)
    for m in range(n-1,2*n-2):
        for k in range(m+1-n,int(np.ceil(m/2.0))+1):
            l = m - k
            sigma[k+2,l+1] = sigma[k+1,l+2] - (a[k+n+1]-a[l])*sigma[k+1,l+1] \
                    - b[k+n+1]*sigma[k,l+1] + b[l]*sigma[k+1,l]
            if m % 2 == 0: # m is even
                a[k+n+1] = a[k] + (sigma[k+1,k+2]-b[k+n+1]*sigma[k,k+1]) \
                        / sigma[k+1,k+1]
            else: # m is odd
                b[k+n+1] = sigma[k+1,k+1] / sigma[k,k]

    # termination
    a[2*n] = a[n-1] - b[2*n]*sigma[n-1,n]/sigma[n,n]

    # matrix T_(2n+1)
    t = np.zeros((2*n+1,2*n+1))
    for i in range(2*n+1):
        t[i, i] = a[i]
    for i in range(1, 2*n+1):
        t[i-1,i] = np.sqrt(b[i])
        t[i,i-1] = np.sqrt(b[i])

    # obtain eigenvalues and eigenvectors of T
    x, v = np.linalg.eig(t)
    
    # weights
    w = v[0,:]**2 * b[0]

    # sort nodes and weights
    sort_index = np.argsort(x)

    return x[sort_index], w[sort_index]

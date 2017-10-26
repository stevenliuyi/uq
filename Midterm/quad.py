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
# reference: Laurie 1997 Calculation of Gauss-Kronrod Quadrature Rules (appendix A)
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
    s = np.zeros(int(np.floor(n/2.0))+2)
    t = np.zeros(int(np.floor(n/2.0))+2)
    t[1] = b[n+1]

    # eastward phase (Salzer's algorithm)
    for m in range(n-1):
        k0 = int(np.floor((m+1)/2.0))
        k = np.arange(int(np.floor((m+1)/2.0)),-1,-1)
        l = m - k
        s[k+1] = np.cumsum((a[k+n+1]-a[l])*t[k+1] + b[k+n+1]*s[k] - b[l]*s[k+1])
        s, t = t, s # swap

    j = int(np.floor(n/2.0)) + 1
    s[1:(j+1)] = s[:j]

    # southward phase (Sack-Donovan-Wheeler algorithm)
    for m in range(n-1,2*n-2):
        k = np.arange(m+1-n,int(np.floor((m-1)/2.0))+1)
        l = m - k
        j = n - 1 - l
        s[j+1] = np.cumsum(-(a[k+n+1]-a[l])*t[j+1] - b[k+n+1]*s[j+1] + b[l]*s[j+2])
        j = j[-1]
        if m % 2 == 0: # even
            k = int(m/2)
            a[k+n+1] = a[k] + (s[j+1] - b[k+n+1]*s[j+2]) / t[j+2]
        else: # odd
            k = int((m+1)/2)
            b[k+n+1] = s[j+1] / s[j+2]
        s, t = t, s
    
    # termination
    a[2*n] = a[n-1] - b[2*n]*s[1]/t[1]

    # matrix T_(2n+1)
    mat_t = np.zeros((2*n+1,2*n+1))
    for i in range(2*n+1):
        mat_t[i, i] = a[i]
    for i in range(1, 2*n+1):
        mat_t[i-1,i] = np.sqrt(b[i])
        mat_t[i,i-1] = np.sqrt(b[i])

    # obtain eigenvalues and eigenvectors of T
    x, v = np.linalg.eig(mat_t)
    
    # weights
    w = v[0,:]**2 * b[0]

    # sort nodes and weights
    sort_index = np.argsort(x)

    return x[sort_index], w[sort_index]

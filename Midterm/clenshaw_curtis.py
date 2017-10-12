import numpy as np

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

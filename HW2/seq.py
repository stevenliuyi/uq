import numpy as np
from sympy import nextprime

# Halton and Faure sequence
class HaltonSeq:
    def __init__(self, n_samples, dim, faure=False, initial_base=3):
        self.n_samples = n_samples
        self.dim = dim
        self.faure = faure
        self.seq = self.generate_halton_seq(initial_base)

    # generate Halton sequence
    # initial_base is the base for the first dimension
    def generate_halton_seq(self, initial_base):

        seq = []
        base = initial_base - 1 # initial value
        for d in range(self.dim):
            base = nextprime(base)
            seq.append([self.vdc(i,base) for i in range(self.n_samples)])
        return np.array(seq).transpose()

    # obtain Faure permutation by recursion
    def faure_permutation(self, base):
        if base == 2: return np.array([0,1])
        
        # even
        if (base % 2 == 0):
            return np.concatenate((2*self.faure_permutation(base/2),
                                   2*self.faure_permutation(base/2)+1))
        # odd
        else:
            k = int((base-1)/2)
            faure = self.faure_permutation(base-1)
            faure += np.array(faure>=k)
            return np.concatenate((faure[:k],[k],faure[k:]))
            
            
    # ith element of van der Corput sequence
    def vdc(self, i, base):
        # for Faure sequence
        if (self.faure): faure = self.faure_permutation(base)

        # initial values
        f = 1
        r = 0
        while i > 0:
            f /= float(base)
            if (self.faure):
                r += f * faure[int(i % base)]
            else:
                r += f * (i % base)
            i = np.floor(i / base)
        return r

    def get(self):
        return self.seq

# Sobol sequence
# reference: http://web.maths.unsw.edu.au/~fkuo/sobol/
class SobolSeq:
    def __init__(self, n_samples, dim, filename='new-joe-kuo-6.21201'):
        self.n_samples = n_samples
        self.dim = dim
        self.filename = filename
        self.seq = self.generate_sobol_seq()

    def get_parameters(self):
        f = open(self.filename, 'r')
        f.readline()

        while (1):
            line = f.readline()
            param = [int(x) for x in line.split()]
            s = param[1] # degree of primitive polynomial
            a = param[2] # number representing the coefficients
            m = param[3:] # initial direction numbers
            yield s, a, m

    def generate_sobol_seq(self):
        # max number of bits needed
        l = int(np.ceil(np.log(self.n_samples) / np.log(2.0)))

        # index of the first 0 digit from the right in binary presentation
        c = np.ones(self.n_samples).astype(int)
        for i in range(1, self.n_samples):
            value = i
            while (value & 1):
                value >>= 1
                c[i] += 1

        points = np.zeros((self.n_samples, self.dim))

        ## first dimension
        # direction numbers scaled by 2^32
        v = [1 << (32-i) for i in range(1, l+1)]

        # evaluate x[0] to x[n-1], scaled by 2^32
        x = np.zeros(self.n_samples).astype(int)
        for i in range(1, self.n_samples):
            x[i] = x[i-1] ^ v[c[i-1]-1]
            # actual points
            points[i, 0] = x[i] / float(2**32)

        ## remaining dimensions
        param_generator = self.get_parameters()
        for j in range(1, self.dim):
            # obtain parameters
            s, a, m = next(param_generator)

            # direction numbers scaled by 2^32
            v = np.zeros(l).astype(int)
            if (l <= s):
                v = [m[i] << (32-i-1) for i in range(l)]
            else:
                v[:s] = [m[i] << (32-i-1) for i in range(s)]
                for i in range(s, l):
                    v[i] = v[i-s] ^ (v[i-s] >> s)
                    for k in range(1, s):
                        v[i] ^= (((a >> (s-1-k)) & 1) * v[i-k])

            # evaluate x[0] to x[n-1], scaled by 2^32
            x = np.zeros(self.n_samples).astype(int)
            for i in range(1, self.n_samples):
                x[i] = x[i-1] ^ v[c[i-1]-1]
                # actual points
                points[i, j] = x[i] / float(2**32)

        return points

    def get(self):
        return self.seq

import numpy as np
from sympy import nextprime

class HaltonSeq:
    def __init__(self, n_samples, dim, faure=False):
        self.n_samples = n_samples
        self.dim = dim
        self.faure = faure
        self.seq = self.generate_halton_seq()

    # generate Halton sequence
    def generate_halton_seq(self):

        seq = []
        base = 2 # initial value
        for d in range(self.dim):
            base = nextprime(base)
            seq.append([self.vdc(i,base) for i in range(self.n_samples)])
        return np.array(seq)

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

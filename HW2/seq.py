import numpy as np
from sympy import nextprime

class HaltonSeq:
    def __init__(self, n_samples, dim):
        self.n_samples = n_samples
        self.dim = dim
        self.seq = self.generate_halton_seq()

    # generate Halton sequence
    def generate_halton_seq(self):
        seq = []
        base = 1 # initial value
        for d in range(self.dim):
            base = nextprime(base)
            seq.append([self.vdc(i,base) for i in range(self.n_samples)])
        return np.array(seq)

    # ith element of van der Corput sequence
    @staticmethod
    def vdc(i, base):
        f = 1
        r = 0
        while i > 0:
            f /= float(base)
            r += f * (i % base)
            i = np.floor(i / base)
        return r

    def get(self):
        return self.seq

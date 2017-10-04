# test sequence generators
import seq
import matplotlib.pyplot as plt

def plot(seq, name):
    # check the first two dimensions
    plt.plot(seq[:,0], seq[:,1], 'o')
    plt.title('%s (first two dimensions)' % name)
    plt.show()
    # check the last two dimensions
    plt.plot(seq[:,10], seq[:,11], 'o')
    plt.title('%s (last two dimensions)' % name)
    plt.show()

# Halton sequence (12 dimensions)
halton = seq.HaltonSeq(1000,12,initial_base=5).get()
plot(halton, 'Halton sequence')

# Faure sequence (12 dimensions)
faure = seq.HaltonSeq(1000,12,faure=True,initial_base=5).get()
plot(faure, 'Faure sequence')

# first 10 Sobol points in 3 dimensions
# (same example as on http://web.maths.unsw.edu.edu/~fkuo/sobol/ )
sobol = seq.SobolSeq(10, 3).get()
print('First 10 Sobol points in 3 dimensions:')
print(sobol)

# Sobol sequence (12 dimensions)
sobol2 = seq.SobolSeq(1000, 12).get()
plot(sobol2, 'Sobol sequence')

# random sequence (12 dimensions)
random = seq.RandomSeq(1000, 12).get()
plot(random, 'random sequence')

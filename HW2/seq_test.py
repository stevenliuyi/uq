# test sequence generators
import seq
import matplotlib.pyplot as plt

# Halton sequence (12 dimensions)
halton = seq.HaltonSeq(1000,12,initial_base=5).get()

# check the first two dimensions
plt.plot(halton[:,0], halton[:,1], 'o')
plt.show()
# check the last two dimensions
plt.plot(halton[:,10], halton[:,11], 'o')
plt.show()

# Faure sequence (12 dimensions)
faure = seq.HaltonSeq(1000,12,faure=True,initial_base=5).get()

# check the first two dimensions
plt.plot(faure[:,0], faure[:,1], 'o')
plt.show()
# check the last two dimensions
plt.plot(faure[:,10], faure[:,11], 'o')
plt.show()


# first 10 Sobol points in 3 dimensions
# (same example as on http://web.maths.unsw.edu.edu/~fkuo/sobol/ )
sobol = seq.SobolSeq(10, 3).get()
print('First 10 Sobol points in 3 dimensions:')
print(sobol)

# Sobol sequence (12 dimensions)
sobol2 = seq.SobolSeq(1000, 12).get()

# check the first two dimensions
plt.plot(sobol2[:,0], sobol2[:,1], 'o')
plt.show()
# check the last two dimensions
plt.plot(sobol2[:,10], sobol2[:,11], 'o')
plt.show()

# random sequence (12 dimensions)
halton = seq.RandomSeq(1000, 12).get()

# check the first two dimensions
plt.plot(halton[:,0], halton[:,1], 'o')
plt.show()
# check the last two dimensions
plt.plot(halton[:,10], halton[:,11], 'o')
plt.show()

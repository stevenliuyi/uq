# test sequence generators
import seq
import matplotlib.pyplot as plt

# Halton sequence (12 dimensions)
halton = seq.HaltonSeq(1000,12).get()

# check the first two dimensions
plt.plot(halton[0,:], halton[1,:], 'o')
plt.show()
# check the last two dimensions
plt.plot(halton[10,:], halton[11,:], 'o')
plt.show()

# Faure sequence (12 dimensions)
faure = seq.HaltonSeq(1000,12,faure=True).get()

# check the first two dimensions
plt.plot(faure[0,:], faure[1,:], 'o')
plt.show()
# check the last two dimensions
plt.plot(faure[10,:], faure[11,:], 'o')
plt.show()

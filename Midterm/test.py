import numpy as np
import matplotlib.pyplot as plt
from clenshaw_curtis import *

# test Clenshaw Curtis rule
points, weights = clenshaw_curtis(-1,1,9)

ex_points = [1.0000000000000000,
             0.9238795325112867,
             0.7071067811865475,
             0.3826834323650897,
             0.0000000000000000,
            -0.3826834323650898,
            -0.7071067811865475,
            -0.9238795325112867,
            -1.0000000000000000]

ex_weights = [0.1587301587301588E-01,
              0.1462186492160182,
              0.2793650793650794,
              0.3617178587204897,
              0.3936507936507936,
              0.3617178587204898,
              0.2793650793650794,
              0.1462186492160182,
              0.1587301587301588E-01]

print('point distance norm: %e' % (np.linalg.norm(points-ex_points, 2)))
print('weight distance norm: %e' % (np.linalg.norm(weights-ex_weights, 2)))

# plot nodes and weights for comparison
plt.plot(points, weights, label='computed')
plt.plot(ex_points, ex_weights, label='benchmark')
plt.title('nodes and weights for the Clenshaw Curtis rule')
plt.legend(loc='lower left')
plt.show()

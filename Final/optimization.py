import sys
import numpy as np
import scipy as sp
from simanneal import Annealer

sys.path.append('../HW2')
import seq

class Optimization:
    def __init__(self, opt_method, n_restarts=100, n_steps=1000):
        if opt_method is None:
            self.opt_method = 'nelder-mead'
        else:
            self.opt_method = opt_method
        self.n_restarts = n_restarts
        self.n_steps = n_steps

    @staticmethod
    def nelder_mead(fun, bounds, n_restarts):
        dim = bounds.shape[0]

        # use Sobol sequence to generate initial values
        initial_values = seq.SobolSeq(n_restarts, dim).get()

        # map [0,1] to bounds
        for d in range(dim):
            initial_values[:,d] = initial_values[:,d]*(bounds[d,1]-bounds[d,0]) \
                    + bounds[d,0]

        ymin = np.inf
        # use Nelder-Mead optimization for each Sobol sample
        for i in range(n_restarts):
            res = sp.optimize.minimize(fun, initial_values[i,:])

            # check if x is within the bounds
            if np.product(res.x<=bounds[:,1]) and np.product(res.x>=bounds[:,0]):
                if (res.fun < ymin):
                    ymin = res.fun
                    x = res.x

        try:
            return x, ymin
        except NameError:
            print('Cannot find the minimum value! Try increase the number of restarts.')
            sys.exit(1)

    @staticmethod
    def anneal(fun, bounds, n_steps):
        dim = bounds.shape[0]

        class SimulatedAnnealer(Annealer):
            def move(self):
                # randomly update state
                for d in range(dim):
                    length = bounds[d,1]-bounds[d,0]
                    self.state[d] += np.random.normal(0,1)*length/10
                    if self.state[d] > bounds[d,1]: self.state[d] = bounds[d,1]
                    if self.state[d] < bounds[d,0]: self.state[d] = bounds[d,0]

            # objective function
            def energy(self):
                return fun(self.state)

        # generate random initial state
        initial_state = np.random.uniform(0, 1, size=dim)
        initial_state = initial_state*(bounds[:,1]-bounds[:,0]) + bounds[:,0]

        sa = SimulatedAnnealer(initial_state)
        sa.steps = n_steps

        # run simulated annealing
        x, ymin = sa.anneal()

        return x, ymin

    @staticmethod
    def lm(fun, bounds, n_restarts):
        dim = bounds.shape[0]

        # use Sobol sequence to generate initial values
        initial_values = seq.SobolSeq(n_restarts, dim).get()

        # map [0,1] to bounds
        for d in range(dim):
            initial_values[:,d] = initial_values[:,d]*(bounds[d,1]-bounds[d,0]) \
                    + bounds[d,0]

        ymin = np.inf
        # use Levenberg–Marquardt algorithm for each Sobol sample
        for i in range(n_restarts):
            res = sp.optimize.least_squares(fun, initial_values[i,:], method='lm')

            # check if x is within the bounds (LM doesn't support bounds)
            if np.product(res.x<=bounds[:,1]) and np.product(res.x>=bounds[:,0]):
                if (res.fun < ymin):
                    ymin = res.fun
                    x = res.x
        try:
            return x, ymin
        except NameError:
            print('Cannot find the minimum value! Try increase the number of restarts.')
            sys.exit(1)

    def optimization(self, fun, bounds):

        if self.opt_method == 'nelder-mead':
            return Optimization.nelder_mead(fun, bounds, self.n_restarts)
        elif self.opt_method == 'anneal':
            return Optimization.anneal(fun, bounds, self.n_steps)
        elif self.opt_method == 'lm':
            return Optimization.lm(fun, bounds, self.n_restarts)
        else:
            print('Cannot find optimization method ' + self.opt_method + '!')
            sys.exit(1)

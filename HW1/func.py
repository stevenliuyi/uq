import numpy as np

# parametric response u1(y)
def u1(y):
    return 0.5 - (y[0]-0.5)**2 - (y[1]-0.5)**2

# parametric response u2(y)
def u2(y):
    delta = 0.1
    return 1/(abs(0.3-y[0]**2-y[1]**2) + delta)

# a control variate for u2(y) 
def u2_control_variate(y):
    return 8 * np.exp(-np.abs(y[0]+y[1]-0.7)*6) + 2

# exact solution for the control variate
u2_cv_expectation = 2*(5+5*np.exp(18/5)-10*np.exp(6)+87*np.exp(39/5))/(45*np.exp(39/5))


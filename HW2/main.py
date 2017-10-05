from quasi_mc import QuasiMC
import numpy as np
import matplotlib.pyplot as plt

# quasi-Monte Carlo estimators
estimators = [QuasiMC('faure'),
              QuasiMC('sobol'),
              QuasiMC('plain')] # add plain Monte Carlo for comparison

# number of samples
n_samples = [10, 100, 1000, 2000, 5000, 10000, 20000, 50000, 100000]

q1 = [] # mean
q2 = [] # variance

# quasi Monte Carlo integration
# loop through estimators
for i in range(len(estimators)):
    q1.append([])
    q2.append([])
    for n in n_samples:
        # quasi Monte Carlo integration
        estimators[i].run(n)
        # print out statistics
        estimators[i].print_results()
        # record statistics
        q1[i].append(estimators[i].mean)
        q2[i].append(estimators[i].variance)

# plots
for i in range(2):
    values = q1 if i == 0 else q2

    # plot statistics v. number of samples
    plt.clf()
    for j in range(len(estimators)):
        plt.plot(n_samples,
                 values[j],
                 label=estimators[j].display_name)
    plt.xscale('log')
    # use symlog here because displacements could be negative
    plt.yscale('symlog') 
    plt.legend(loc='lower right' if i == 0 else 'upper right')
    plt.savefig('Q%d.png' % (i+1), bbox_inches='tight')
    print('statistics v. n plot for Q%d saved!' % (i+1))

    # plot relative rate of change v. number of samples
    plt.clf()
    for j in range(len(estimators)):
        plt.plot(n_samples,
                 # numpy version >= 1.13 required to compute gradient
                 abs(np.gradient(values[j], n_samples) / values[j]),
                 label=estimators[j].display_name)
    plt.xscale('log')
    plt.yscale('log') 
    plt.legend(loc='lower left')
    plt.savefig('Q%d_relative_rate_of_change.png' % (i+1), bbox_inches='tight')
    print('relative rate of change v. n plot for Q%d saved!' % (i+1))

# save statistics
np.save('q1.npy', q1)
np.save('q2.npy', q2)

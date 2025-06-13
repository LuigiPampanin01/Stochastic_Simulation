
# Write a discrete event simulation program for a blocking system,
#  i.e. a system with m service units and no waiting room.
#  The offered traffic A is the product of the mean arrival rate and the mean service time.

# The arrival process is modelled as a Poisson process.
#  Report the fraction of blocked customers, and a confidence interval for this fraction. 
#  Choose the service time distribution as exponential. Parameters: m = 10, mean service time = 8 time units,
#  mean time between customers = 1 time unit (corresponding to an offered traffic of 8 Erlang),
#  10 x 10.000 customers. This system is sufficiently simple such that the analytical solution is known.
#  See the last slide for the solution. Verify your simulation program using this knowledge.
#%%
import numpy as np
import math
from scipy.stats import t
import matplotlib.pyplot as plt

# Parameters 
m = 10
mst = 8
mtbc = 1

n_sim = 10
block_fractions = np.zeros(n_sim)

for j in range(n_sim):
    n_custom = 10000
    n = 0

    interval_between_arrivals = np.random.exponential(scale = mtbc, size = n_custom)
    arrival_times = np.cumsum(interval_between_arrivals)

    service_in_use_times = np.zeros(m)

    n_cust_blocked = 0


    while(n < n_custom):

        available_services = np.where(arrival_times[n] >= service_in_use_times)[0]
        if (len(available_services)>0):
            i = available_services[0]
            service_time = np.random.exponential(scale=mst)
            service_in_use_times[i] = arrival_times[n] + service_time
        else:
            n_cust_blocked += 1
        n += 1

    fraction = n_cust_blocked / n_custom  * 100
    print(f'fraction of blocks: {fraction}%')
    block_fractions[j] = fraction

# Exact solution
A = mst * mtbc
m_vect = np.arange(0,m+1)
denominator = np.sum(A**m_vect / [math.factorial(k) for k in m_vect])
B = A**m/math.factorial(m) / denominator

print(f'exact solution: {B*100}%')

# Confidence interval

theta_hat = np.sum(block_fractions)/n_sim
#print(theta_hat)
sigma_2 = (np.sum(block_fractions ** 2) - n_sim * theta_hat**2) / (n_sim -1)
print(sigma_2)  

dof = n_sim -1
alpha = 0.95
t_quant = t.ppf(1-alpha/2,dof)
CI = [theta_hat - np.sqrt(sigma_2)/np.sqrt(n)*t_quant, theta_hat + np.sqrt(sigma_2)/np.sqrt(n)*t_quant]

print(f'Confidence interval: {CI}')

#%%
# Point 2
# . The arrival process is modelled as a renewal process using the same parameters as in Part 1 when possible.
#  Report the fraction of blocked customers, and a confidence interval for this fraction for at least the following two cases
#  (a) Experiment with Erlang distributed inter arrival times The Erlang distribution should have a mean of 1 (b)
#  hyper exponential inter arrival times. The parameters for the hyper exponential distribution should be
#  p1 = 0.8,λ1 = 0.8333,p2 = 0.2,λ2 = 5.0.

# (a)

k = 10
print(f'Point 2 simulation with Erlang distribution, k = {k}')

# Parameters 
m = 10
mst = 8
mtbc = 1

n_sim = 10
block_fractions = np.zeros(n_sim)

for j in range(n_sim):
    n_custom = 10000
    n = 0

    interval_between_arrivals = np.random.gamma(shape = k, scale = 1/k, size = n_custom)
    arrival_times = np.cumsum(interval_between_arrivals)

    service_in_use_times = np.zeros(m)

    n_cust_blocked = 0


    while(n < n_custom):

        available_services = np.where(arrival_times[n] >= service_in_use_times)[0]
        if (len(available_services)>0):
            i = available_services[0]
            service_time = np.random.exponential(scale=mst)
            service_in_use_times[i] = arrival_times[n] + service_time
        else:
            n_cust_blocked += 1
        n += 1

    fraction = n_cust_blocked / n_custom  * 100
    print(f'fraction of blocks: {fraction}%')
    block_fractions[j] = fraction


# Confidence interval

theta_hat = np.sum(block_fractions)/n_sim
#print(theta_hat)
sigma_2 = (np.sum(block_fractions ** 2) - n_sim * theta_hat**2) / (n_sim -1)
print(sigma_2)  

dof = n_sim -1
alpha = 0.95
t_quant = t.ppf(1-alpha/2,dof)
CI = [theta_hat - np.sqrt(sigma_2)/np.sqrt(n)*t_quant, theta_hat + np.sqrt(sigma_2)/np.sqrt(n)*t_quant]

print(f'Confidence interval: {CI}')

# Graphical comparion of densities
k_values = [1,2,5,10]

samples_exp = np.random.exponential(scale = mtbc, size = n_custom)

plt.figure(figsize=(10, 6))
for k in k_values:
    samples_erlang = np.random.gamma(shape=k, scale=1/k, size=n_custom)
    counts, bins = np.histogram(samples_erlang, bins=100, density=True)
    centers = (bins[:-1] + bins[1:]) / 2
    plt.plot(centers, counts, label=f"Erlang(k={k})")
counts2, bins2 = np.histogram(samples_exp, bins=100, density=True)

centers2 = (bins2[:-1] + bins2[1:]) / 2

plt.plot(centers2, counts2, label="Exponential")
plt.title("Comparison of Erlang and Exponential Densities")
plt.xlabel("x")
plt.legend()
plt.grid(True)
plt.show()

#%%
# (b) Hyperexponential 

p1 = 0.8
p2 = 0.2
lam1 = 0.8333
lam2 = 5.0

print(f'Point 2 simulation with Hyperexponential distribution')

# Parameters 
m = 10
mst = 8
mtbc = 1

n_sim = 10
block_fractions = np.zeros(n_sim)

for j in range(n_sim):
    n_custom = 10000
    n = 0

    exps = np.random.choice([0,1], p = [p1,p2], size = n_custom)
    interval_between_arrivals[exps == 0] = np.random.exponential(scale=1/lam1, size=np.sum(exps == 0))
    interval_between_arrivals[exps == 1] = np.random.exponential(scale=1/lam2, size=np.sum(exps == 1))
    arrival_times = np.cumsum(interval_between_arrivals)

    service_in_use_times = np.zeros(m)

    n_cust_blocked = 0


    while(n < n_custom):

        available_services = np.where(arrival_times[n] >= service_in_use_times)[0]
        if (len(available_services)>0):
            i = available_services[0]
            service_time = np.random.exponential(scale=mst)
            service_in_use_times[i] = arrival_times[n] + service_time
        else:
            n_cust_blocked += 1
        n += 1

    fraction = n_cust_blocked / n_custom  * 100
    print(f'fraction of blocks: {fraction}%')
    block_fractions[j] = fraction


# Confidence interval

theta_hat = np.sum(block_fractions)/n_sim
#print(theta_hat)
sigma_2 = (np.sum(block_fractions ** 2) - n_sim * theta_hat**2) / (n_sim -1)
print(sigma_2)  

dof = n_sim -1
alpha = 0.95
t_quant = t.ppf(1-alpha/2,dof)
CI = [theta_hat - np.sqrt(sigma_2)/np.sqrt(n)*t_quant, theta_hat + np.sqrt(sigma_2)/np.sqrt(n)*t_quant]

print(f'Confidence interval: {CI}')

#Compare the two distributions
samples_exp = np.random.exponential(scale=mtbc, size=n_custom)
choices = np.random.choice([0, 1], size=n_custom, p=[p1, p2])
samples_hyper = np.zeros(n_custom)
samples_hyper[choices == 0] = np.random.exponential(scale=1/lam1, size=np.sum(choices == 0))
samples_hyper[choices == 1] = np.random.exponential(scale=1/lam2, size=np.sum(choices == 1))

plt.figure(figsize=(10, 6))
counts_exp, bins_exp = np.histogram(samples_exp, bins=100, density=True)
centers_exp = (bins_exp[:-1] + bins_exp[1:]) / 2
plt.plot(centers_exp, counts_exp, label="Exponential")

counts_hyper, bins_hyper = np.histogram(samples_hyper, bins=100, density=True)
centers_hyper = (bins_hyper[:-1] + bins_hyper[1:]) / 2
plt.plot(centers_hyper, counts_hyper, label="Hyperexponential")

plt.title("Comparison of Exponential and Hyperexponential Densities")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()
#%%
# Point 3
# Arrival process still a Poisson process, change service time distribution

# (a) Constant service time

print(f'Point 3: constant time service')

for j in range(n_sim):
    n_custom = 10000
    n = 0

    interval_between_arrivals = np.random.exponential(scale = mtbc, size = n_custom)
    arrival_times = np.cumsum(interval_between_arrivals)

    service_in_use_times = np.zeros(m)

    n_cust_blocked = 0


    while(n < n_custom):

        available_services = np.where(arrival_times[n] >= service_in_use_times)[0]
        if (len(available_services)>0):
            i = available_services[0]
            #service_time = np.random.exponential(scale=mst)
            service_in_use_times[i] = arrival_times[n] + mst
        else:
            n_cust_blocked += 1
        n += 1

    fraction = n_cust_blocked / n_custom  * 100
    print(f'fraction of blocks: {fraction}%')
    block_fractions[j] = fraction


# Confidence interval

theta_hat = np.sum(block_fractions)/n_sim
#print(theta_hat)
sigma_2 = (np.sum(block_fractions ** 2) - n_sim * theta_hat**2) / (n_sim -1)
print(sigma_2)  

dof = n_sim -1
alpha = 0.95
t_quant = t.ppf(1-alpha/2,dof)
CI = [theta_hat - np.sqrt(sigma_2)/np.sqrt(n)*t_quant, theta_hat + np.sqrt(sigma_2)/np.sqrt(n)*t_quant]

print(f'Confidence interval: {CI}')

#%%
# (b) Pareto distribution
from scipy.stats import pareto

print(f'Point 3: Pareto distributed time services')

k = 1.05
#mst = E[X] = k /( k-1)*b => b = (k-1)*mst/k
b = (k-1)*mst/k

for j in range(n_sim):
    n_custom = 10000
    n = 0

    interval_between_arrivals = np.random.exponential(scale = mtbc, size = n_custom)
    arrival_times = np.cumsum(interval_between_arrivals)

    service_in_use_times = np.zeros(m)

    n_cust_blocked = 0


    while(n < n_custom):

        available_services = np.where(arrival_times[n] >= service_in_use_times)[0]
        if (len(available_services)>0):
            i = available_services[0]
            service_time = pareto.rvs(k)*b
            service_in_use_times[i] = arrival_times[n] + service_time
        else:
            n_cust_blocked += 1
        n += 1

    fraction = n_cust_blocked / n_custom  * 100
    print(f'fraction of blocks: {fraction}%')
    block_fractions[j] = fraction


# Confidence interval

theta_hat = np.sum(block_fractions)/n_sim
#print(theta_hat)
sigma_2 = (np.sum(block_fractions ** 2) - n_sim * theta_hat**2) / (n_sim -1)
print(sigma_2)  

dof = n_sim -1
alpha = 0.95
t_quant = t.ppf(1-alpha/2,dof)
CI = [theta_hat - np.sqrt(sigma_2)/np.sqrt(n)*t_quant, theta_hat + np.sqrt(sigma_2)/np.sqrt(n)*t_quant]

print(f'Confidence interval: {CI}')

#%%
# Comparing the two pareto distributions we notice that, for k = 1.05, the distribution
# presents way lower values with some sporadic high value, but not enough to balance the other low values
# therefore, the times to serve a client a way lower, resulting in fewer blocks

# Whereas, for k = 2.05, we get much higher values for the distribution, which lead to higher service time
# and therefore a higher number of blocks
# %%

# Uniform distribution 
print(f'Point 3: uniform distribution between 0 and 10 for service time')


for j in range(n_sim):
    n_custom = 10000
    n = 0

    interval_between_arrivals = np.random.exponential(scale = mtbc, size = n_custom)
    arrival_times = np.cumsum(interval_between_arrivals)

    service_in_use_times = np.zeros(m)

    n_cust_blocked = 0


    while(n < n_custom):

        available_services = np.where(arrival_times[n] >= service_in_use_times)[0]
        if (len(available_services)>0):
            i = available_services[0]
            service_time = np.random.uniform(0,2*mst)
            service_in_use_times[i] = arrival_times[n] + service_time
        else:
            n_cust_blocked += 1
        n += 1

    fraction = n_cust_blocked / n_custom  * 100
    print(f'fraction of blocks: {fraction}%')
    block_fractions[j] = fraction


# Confidence interval

theta_hat = np.sum(block_fractions)/n_sim
#print(theta_hat)
sigma_2 = (np.sum(block_fractions ** 2) - n_sim * theta_hat**2) / (n_sim -1)
print(sigma_2)  

dof = n_sim -1
alpha = 0.95
t_quant = t.ppf(1-alpha/2,dof)
CI = [theta_hat - np.sqrt(sigma_2)/np.sqrt(n)*t_quant, theta_hat + np.sqrt(sigma_2)/np.sqrt(n)*t_quant]

print(f'Confidence interval: {CI}')




# %%

# Gaussian Distribution
print(f'Point 3: normal distribution for service time')


for j in range(n_sim):
    n_custom = 10000
    n = 0

    interval_between_arrivals = np.random.exponential(scale = mtbc, size = n_custom)
    arrival_times = np.cumsum(interval_between_arrivals)

    service_in_use_times = np.zeros(m)

    n_cust_blocked = 0


    while(n < n_custom):

        available_services = np.where(arrival_times[n] >= service_in_use_times)[0]
        if (len(available_services)>0):
            i = available_services[0]
            service_time = np.random.normal(mst,3)
            service_in_use_times[i] = arrival_times[n] + service_time
        else:
            n_cust_blocked += 1
        n += 1

    fraction = n_cust_blocked / n_custom  * 100
    print(f'fraction of blocks: {fraction}%')
    block_fractions[j] = fraction


# Confidence interval

theta_hat = np.sum(block_fractions)/n_sim
#print(theta_hat)
sigma_2 = (np.sum(block_fractions ** 2) - n_sim * theta_hat**2) / (n_sim -1)
print(sigma_2)  

dof = n_sim -1
alpha = 0.95
t_quant = t.ppf(1-alpha/2,dof)
CI = [theta_hat - np.sqrt(sigma_2)/np.sqrt(n)*t_quant, theta_hat + np.sqrt(sigma_2)/np.sqrt(n)*t_quant]

print(f'Confidence interval: {CI}')
#%%
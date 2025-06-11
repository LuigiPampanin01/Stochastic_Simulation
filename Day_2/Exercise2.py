#Choose a value for the probability parameter p in the geometric distribution
#and simulate 10,000 outcomes. You can experiment with a small, moderate and large value if you like.


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare
# Point 1

# 10.000 random values from U(0, 1)
n = 10000
U = np.random.uniform(0, 1, size=n)

# value for probability p
p = 0.3
X = np.floor(np.log(U)/(np.log(1-p))) + 1

# True probability
values = np.arange(1, int(X.max()) + 1)
counts = np.array([(X == k).sum() for k in values])

prob_teorica = p * (1 - p) ** (values - 1)
expected_counts = prob_teorica * n


# Generating histogram
plt.figure(figsize=(8, 5))
plt.hist(X, bins=np.arange(1, max(X)+2)-0.5, density=True, edgecolor='black', align='mid')
plt.plot(values, prob_teorica, 'ro-', label='Teorica')
plt.title(f'Simulated geometric distribution (p = {p})')
plt.xlabel('X')
plt.ylabel('Frequence')
plt.grid(True)
plt.show()

# Chi squared test

# Check that the frequency is at least 5
while expected_counts[-1] < 5:
    expected_counts[-2] += expected_counts[-1]
    counts[-2] += counts[-1]
    expected_counts = expected_counts[:-1]
    counts = counts[:-1]
    values = values[:-1]

# Ensure the counts match
expected_counts = expected_counts * (counts.sum() / expected_counts.sum())

chi2_stat, p_value = chisquare(f_obs=counts, f_exp=expected_counts)
print(f"Chi2 stat: {chi2_stat:.4f}, p-value: {p_value:.4f}")

#%%
# Point 2
# (a) Direct method

p = [7/48, 5/48, 1/8, 1/16, 1/4, 5/16]
classes = [1, 2, 3, 4, 5, 6]
cdf = np.cumsum(p)
cdf_dict = dict(zip(classes, cdf))

indices = np.searchsorted(cdf, U)
X_samples = np.array(classes)[indices]
counts = np.array([(X_samples == c).sum() for c in classes])

# If we only use one point:
# u = np.random.uniform(0,1)
#for i, F in enumerate(cdf):
#    if u <= F:
#        X = classes[i]
#        break

expected_values = np.array(n * np.array(p))

# Generating histogram
plt.figure(figsize=(8, 5))
plt.hist(X_samples, bins=np.arange(1, max(X)+2)-0.5, density=True, edgecolor='black', align='mid')
plt.plot(classes, p, 'ro-', label='Teorica')
plt.title(f'Simulated discrete distribution')
plt.xlabel('X')
plt.ylabel('Frequence')
plt.grid(True)
plt.show()


# Chi squared test
# Check that the frequency is at least 5
while expected_values[-1] < 5:
    expected_values[-2] += expected_values[-1]
    counts[-2] += counts[-1]
    expected_values = expected_values[:-1]
    counts = counts[:-1]
    classes = classes[:-1]

# Ensure the counts match
expected_values = expected_values * (np.array(values).sum() / np.array(expected_values).sum())

chi2_stat, p_value = chisquare(f_obs=values, f_exp=expected_values)
print(f"Chi2 stat: {chi2_stat:.4f}, p-value: {p_value:.4f}")


# %%

# Rejection method

import numpy as np
import matplotlib.pyplot as plt

# Define the exponential distribution parameters
lam = 0.5  # lambda parameter

# Generate values for x-axis (time)
#x = np.linspace(0, 10, 1000)
x1 = np.linspace(0.01, 1, 1000)
#x2 = np.linspace(0, 1, 1000)

# Calculate the cumulative distribution function (CDF) of the exponential distribution
#cdf = 1 - np.exp(-lam * x)
cdf1 = x1 - x1 * np.log(x1)
#cdf2 = x2 + (1 - x2) * np.log(1-x2)

# Plot the CDF
#lt.plot(x, cdf, color='purple')
plt.plot(x1*100, cdf1*100, color='purple')
#plt.plot(x, cdf2, color='green')
plt.xlabel('% of Assortment')
plt.ylabel('% of Demand')
plt.title('Demand curve for exponential distribution (Lamballais, 2019)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(np.arange(0, 101, 10))
plt.yticks(np.arange(0, 101, 10))
#plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.savefig('Demand_Curves_exp_lorenz.png')
plt.show()

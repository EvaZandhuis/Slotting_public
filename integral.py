from scipy.optimize import fsolve
from scipy.integrate import quad

# Define the functions
def f1(i):
    return 0.222 * i**(0.222 - 1)

def f2(i):
    return 0.139 * i**(0.139 - 1)

def f3(i):
    return 0.065 * i**(0.065 - 1)

# Define the integral equation
def integral_eqn(bounds, func):
    a, b = bounds
    integral, _ = quad(func, a, b)
    return [integral - 1]

# Solve for the bounds
bounds_f1 = fsolve(integral_eqn, [0.01, 10], args=(f1,))
bounds_f2 = fsolve(integral_eqn, [0.01, 10], args=(f2,))
bounds_f3 = fsolve(integral_eqn, [0.01, 10], args=(f3,))

# Print the results
print("Bounds for f1:", bounds_f1)
print("Bounds for f2:", bounds_f2)
print("Bounds for f3:", bounds_f3)



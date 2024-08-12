from sympy import symbols, Eq, solve
from sympy.abc import x, y

# Define variables
#h, v = symbols('h v')

# Define equations
sol = solve(0.065 * (0.05375 +1) ** -0.935 , x, dict=True)
print(sol)




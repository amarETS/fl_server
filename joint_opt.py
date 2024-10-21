import cvxpy as cp
import numpy as np

# Define the parameters
M = 5  # Number of participants (example)
T = 10  # Number of time periods (example)
B = 10  # Total available bandwidth
p_tr = 0.5  # Transmission power
p_c = 0.2  # Computational power
D_m = np.random.rand(M) * 100  # Data size for each participant (example)
c_m = np.random.rand(M) * 10  # Computational complexity factor
f_min = 0.1  # Minimum frequency
S_m = np.random.rand(M) * 100  # Size of each task (example)
rho = 0.5  # Penalty factor
K_l = 1.0  # Some constant (example)
K_epsilon = 1.0  # Some constant (example)
b_min = 0.1  # Minimum value for b_m^t

# Define variables
a_t = cp.Variable((M, T), boolean=True)  # a_m^t binary variable
b_t = cp.Variable((M, T))  # b_m^t continuous variable
f_m = cp.Variable(M)  # Frequency variables for each participant

# Define cost function (equation {cost(t)})
def cost_function(a_t, b_t, f_m):
    transmission_cost = cp.sum(cp.multiply(a_t, cp.multiply(b_t, B * p_tr)))
    computation_cost = cp.sum(cp.multiply(a_t, cp.multiply(D_m, c_m / f_m) * p_c))
    
    delay_term_1 = cp.sum(cp.multiply(S_m, 1 / (b_t * B)))
    delay_term_2 = cp.max(cp.multiply(S_m, 1 / (b_t * B)))

    total_cost = K_epsilon * transmission_cost + computation_cost + rho * (delay_term_1 + K_l * delay_term_2)
    return total_cost

# Constraints
constraints = []
for t in range(T):
    # Bandwidth constraint (equation \ref{opt1:a})
    constraints.append(cp.sum(a_t[:, t] * b_t[:, t] * B) <= B)

    # Sum of b_m^t constraint (equation \ref{opt1:b})
    constraints.append(cp.sum(b_t[:, t]) == 1)
    
    # Bound on b_m^t (equation \ref{opt1:c})
    constraints.append(b_t[:, t] >= b_min)
    constraints.append(b_t[:, t] <= 1)

# Binary constraint for a_m^t (equation \ref{opt1:d}) is handled by using boolean=True in the variable definition

# Frequency constraint (equation \ref{opt1:e})
constraints.append(f_m >= f_min)

# Add any other relevant constraints, for example:
# Add constraint (equation \ref{opt1:f}) and (equation \ref{opt1:g})

# Objective
objective = cp.Minimize(cost_function(a_t, b_t, f_m))

# Solve the problem
problem = cp.Problem(objective, constraints)
problem.solve()

# Output the solution
print("Optimal cost:", problem.value)
print("Optimal a_t:", a_t.value)
print("Optimal b_t:", b_t.value)
print("Optimal frequencies:", f_m.value)

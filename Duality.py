import numpy as np
from scipy.optimize import linprog

c_primal = [-1, -4, -2]
A_ub_primal = np.array([
    [5, 2, 2],
    [4, 8, -8],
    [1, 1, 4]
])
b_ub_primal = [145, 260, 190]

result_primal = linprog(c_primal, A_ub=A_ub_primal, b_ub=b_ub_primal, bounds=(0, None), method='highs')
x_primal = result_primal.x
z_primal = -result_primal.fun

print(f"\nPrimal problem : \n {A_ub_primal} ")
print("Primal Solution:")
print(f"x = {x_primal}")
print(f"Objective value (z) = {z_primal}")

Q = np.array([0, 52.5, 20])
feasible = np.all(np.dot(A_ub_primal, Q) <= b_ub_primal) and np.all(Q >= 0)
print(f"\nIs {Q} feasible for the primal problem?")
print("Yes" if feasible else "No")

c_dual = [145, 260, 190] #equals b_ub_primal
A_ub_dual = np.array(A_ub_primal).T*-1

b_ub_dual = [-1, -4, -2] #equals c_primal

result_dual = linprog(c_dual, A_ub=A_ub_dual, b_ub=b_ub_dual, bounds=(0, None), method='highs')
y_dual = result_dual.x
w_dual = result_dual.fun

print(f"\nDual problem : \n {A_ub_dual} ")
print("\nDual Solution :")
print(f"y = {y_dual}")
print(f"Objective value (w) = {w_dual}")

if np.isclose(z_primal, w_dual):
    print(f"Primal and dual objectives match, so {Q} is the solution for the primal problem")
else:
    print(f"Strong duality does not hold, so {Q} is not the solution for the primal problem")


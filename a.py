from numpy.random import randn
import cvxpy as cp
from qcqp import *
import numpy as np


def solve(K_a, K_b, mu_a, mu_b, dims):

    K_a = np.matrix.round(K_a, decimals= 3)
    n = 2
    m = 2
    p = 2
    np.random.seed(1)
    P = 1000000000000000* np.random.randn(n, n)
    P = P.T @ P
    q = np.random.randn(n)
    G = np.random.randn(m, n)
    h = G @ np.random.randn(n)
    A = np.random.randn(p, n)
    b = np.random.randn(p)
    # mu_a = np.random.rand(n)
    print("----------_P_-------------------")
    print(P)
    print(np.linalg.eig(P)[0])
    print("-----------_KA_-----------------")
    print(K_a)
    print(np.linalg.eig(K_a)[0])

    x = cp.Variable(n)
    j = cp.Constant(K_a - K_b)
    obj = cp.sum_squares(mu_a - x)
    t = cp.quad_form(x, K_a) - cp.quad_form(x, K_b) + 2*(mu_a.T@K_a - mu_b.T@K_b)*x - mu_b.T @ K_b @ mu_b - mu_a.T @ K_a @ mu_a
    
    print("Constraint" + t.curvature)
    print("Objective" + obj.curvature)
    # Define and solve the CVXPY problem.
    prob = cp.Problem(cp.Minimize(obj), [t == 0])
    prob.solve()


    # Print result.
    print("\nThe optimal value is", prob.value)
    print("A solution x is")
    print(x.value)

# from numpy.random import randn
# import cvxpy as cp
# from qcqp import *
# import numpy as np


# def solve(K_a, K_b, mu_a, mu_b, dims):

#     P = np.copy(K_a)
#     m = cp.Constant(K_a)
#     print(P)

#     # Define and solve the CVXPY problem.
#     x = cp.Variable(2)
#     t = mu_a - x
#     # mu_a
#     obj = cp.quad_form(t, m)
#     print(t.curvature)
#     print(obj.curvature)
#     print(m.curvature)

#     prob = cp.Problem(cp.Minimize(obj),
#                     [cp.quad_form(x, K_a-K_b) + 2*(mu_a.T@K_a - mu_b.T@K_b)*x == mu_b.T @ K_b @ mu_b - mu_a.T @ K_a @ mu_a])
#     prob.solve()


#     # Print result.
#     print("\nThe optimal value is", prob.value)
#     print("A solution x is")
#     print(x.value)
#     print("A dual solution corresponding to the inequality constraints is")
#     print(prob.constraints[0].dual_value)


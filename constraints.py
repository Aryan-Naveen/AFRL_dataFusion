import numpy as np
import numpy.linalg as LA
from scipy.linalg import sqrtm
from numpy.linalg import det

def inv(A):
    return LA.inv(A)

def mutual_covariance(cov_a, cov_b):
    D_a, S_a = np.linalg.eigh(cov_a)
    D_a_sqrt = sqrtm(np.diag(D_a))
    D_a_sqrt_inv = inv(D_a_sqrt)
    M = np.dot(np.dot(np.dot(np.dot(D_a_sqrt_inv, inv(S_a)), cov_b), S_a), D_a_sqrt_inv)    # eqn. 10 in Sijs et al.
    D_b, S_b = np.linalg.eigh(M)
    D_gamma = np.diag(np.clip(D_b, a_min=1.0, a_max=None))   # eqn. 11b in Sijs et al.
    return np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(S_a, D_a_sqrt), S_b), D_gamma), inv(S_b)), D_a_sqrt), inv(S_a))  # eqn. 11a in Sijs et al


class Constraints():
    def __init__(self, C_a, C_b, x_a, x_b, eta):
        self.created = True
        self.C_a = C_a
        self.C_b = C_b
        self.x_a = x_a
        self.x_b = x_b
        self.eta = eta

    def objective(self, S):
        S = S.reshape(1, 2)
        return np.trace(S.T @ S)

    def constraint1(self, S):
        S = S.reshape(1, 2)    
        A = S.T @ S - 1e-10*np.identity(2)
        return np.linalg.eig(A)[0][0]

    def constraint2(self, S):
        S = S.reshape(1, 2)    
        A = S.T @ S - 1e-10*np.identity(2)
        return np.linalg.eig(A)[0][1]

    def prob_constraint(self, S):
        S = S.reshape(1, 2)    
        C_c_inv = LA.inv(mutual_covariance(self.C_a, self.C_b) + 1e-10*np.identity(2) + S.T @ S)

        C_ac = LA.inv(inv(self.C_a) - C_c_inv)
        C_bc = LA.inv(inv(self.C_b) - C_c_inv)

        C_abc_inv_inv = LA.inv(LA.inv(C_ac) + LA.inv(C_bc))
        C_abc_inv = LA.inv(C_ac + C_bc)

        x_c = (C_abc_inv_inv @ (LA.inv(C_ac) @ self.x_a.T + LA.inv(C_bc) @ self.x_b.T)).T
        
        x_ac = (C_ac @ (inv(self.C_a) @ self.x_a.T - C_c_inv @ x_c.T)).T
        x_bc = (C_bc @ (inv(self.C_b) @ self.x_b.T - C_c_inv @ x_c.T)).T
        
        f = ((x_ac - x_bc) @ LA.inv(C_ac+C_bc) @ (x_ac - x_bc).T)[0][0]
        return self.eta - f
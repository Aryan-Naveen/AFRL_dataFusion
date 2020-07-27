import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2
from scipy.linalg import sqrtm
from numpy.linalg import det
import numpy.linalg as LA


def find_direction(data):
    
    C_a = data.get_C_a()
    C_b = data.get_C_b()

    x_a = data.get_x_a()
    x_b = data.get_x_b()

    x_a = x_a.reshape(1, 2)
    x_b = x_b.reshape(1, 2)

    def mutual_covariance(cov_a, cov_b):
        D_a, S_a = np.linalg.eigh(cov_a)
        D_a_sqrt = sqrtm(np.diag(D_a))
        D_a_sqrt_inv = inv(D_a_sqrt)
        M = np.dot(np.dot(np.dot(np.dot(D_a_sqrt_inv, inv(S_a)), cov_b), S_a), D_a_sqrt_inv)    # eqn. 10 in Sijs et al.
        D_b, S_b = np.linalg.eigh(M)
        D_gamma = np.diag(np.clip(D_b, a_min=1.0, a_max=None))   # eqn. 11b in Sijs et al.
        return np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(S_a, D_a_sqrt), S_b), D_gamma), inv(S_b)), D_a_sqrt), inv(S_a))  # eqn. 11a in Sijs et al

    def get_critical_value(dimensions, alpha):
        return chi2.ppf((1 - alpha), df=dimensions)

    eta = get_critical_value(2, 0.05)

    def inv(mat):
        return np.linalg.inv(mat)

    def constraint1(theta):
        return np.pi - theta[0]

    def rotation_matrix(theta):
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[c, s], [-s, c]])

    def objective(theta):
        v_0 = (x_a - x_b)/np.linalg.norm(x_a - x_b)
        R = rotation_matrix(theta)
        
        v = (R @ v_0.T).T
        if np.linalg.det(v.T@v) < 0:
            v *= -1

        C_c_inv = inv(mutual_covariance(C_a, C_b) + 10*v.T @ v)
        
        C_ac_inv = inv(C_a) - C_c_inv
        C_ac = inv(C_ac_inv)

        C_bc_inv = inv(C_b) - C_c_inv
        C_bc = inv(C_bc_inv)

        x_c = (inv(C_ac_inv + C_bc_inv) @ (C_ac_inv @ x_a.T + C_bc_inv @ x_b.T - C_c_inv@(x_a + x_b).T)).T
        x_ac = (C_ac @ (inv(C_a) @ x_a.T - C_c_inv @ x_c.T)).T
        x_bc =(C_bc @ (inv(C_b) @ x_b.T - C_c_inv @ x_c.T)).T
        f = abs(((x_ac - x_bc) @ inv(C_ac + C_bc) @ (x_ac - x_bc).T)[0][0])
        return f

    theta = np.linspace(0, 2*np.pi, 1024)

    obj = np.vectorize(objective)

    f = obj(theta)

    t = f[np.argwhere(f==np.min(f))][0][0]
    g = f[np.argwhere(f==np.max(f))][0][0]

    v_0 = (x_a - x_b)/np.linalg.norm(x_a - x_b)
    
    return (rotation_matrix(t) @ v_0.T).T, (rotation_matrix(g) @ v_0.T).T
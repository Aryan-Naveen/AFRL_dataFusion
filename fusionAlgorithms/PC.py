from tqdm import tqdm
import numpy as np

def inv(A):
    return np.linalg.inv(A)

def calculate_MSE(C_a, C_b, C_c, x_a, x_b, C_fus):
    C_ac_inv = inv(C_a) - inv(C_c)
    C_ac = inv(C_ac_inv)
    C_bc_inv = inv(C_b) - inv(C_c)
    C_bc = inv(C_bc_inv)
    C_c_inv = inv(C_c)
    
    
    C_abc_inv_inv = inv(C_ac_inv + C_bc_inv)
    
    
    x_c = (C_abc_inv_inv @ (C_ac_inv @ x_a.T + C_bc_inv @ x_b.T)).T
    x_ac = (C_ac @ (inv(C_a) @ x_a.T - C_c_inv @ x_c.T)).T
    x_bc =(C_bc @ (inv(C_b) @ x_b.T - C_c_inv @ x_c.T)).T
    
    x_fus = C_fus @ (C_ac_inv @ x_ac.T + C_bc_inv @ x_bc.T + C_c_inv @ x_c.T)
    
    mse = (np.square(true_x_fus - x_fus)).mean()
    return x_fus
    
    

def fusion(x_a, x_b, C_a, C_b):
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    from scipy.optimize import minimize
    from scipy.stats import chi2
    from scipy.linalg import sqrtm
    from numpy.linalg import det
    import numpy.linalg as LA
    import matplotlib.pyplot as plt
    import math

    debug= False

    def generate_covariance(true_mu, dims):
        S = np.tril(np.random.randn(dims, dims))
        cov = np.dot(S, S.T)
        while(abs(np.linalg.det(cov)) < 1.5):
            cov = cov + 0.5*np.diag(np.diag(cov))
        mu = np.random.multivariate_normal(true_mu, cov, 1)[0]

        return mu, cov


    def plot_ellipse(covariance, ax, label_t="", linestyle='', alpha_val=0.25, color_def='red', center = [0, 0]):
        if covariance.shape[0] == 2:
            x_el = np.array([np.sin(np.linspace(0, 2*math.pi, num=63)), np.cos(np.linspace(0, 2*math.pi, num=63))])
            C = np.linalg.cholesky(covariance)
            y_el = np.dot(C, x_el)
            if len(linestyle) > 0:
                if len(label_t) > 0:
                    ax.plot(y_el[0] + center[0], y_el[1] + center[1], label=label_t, alpha=alpha_val, color=color_def, linestyle=linestyle)
                else:
                    ax.plot(y_el[0] + center[0], y_el[1] + center[1], alpha=alpha_val, color=color_def, linestyle=linestyle)            
            else:
                if len(label_t) > 0:
                    ax.plot(y_el[0] + center[0], y_el[1] + center[1], label=label_t, alpha=alpha_val, color=color_def)
                else:
                    ax.plot(y_el[0] + center[0], y_el[1] + center[1], alpha=alpha_val, color=color_def)            

    def mutual_covariance(cov_a, cov_b):
        D_a, S_a = np.linalg.eigh(cov_a)
        D_a_sqrt = sqrtm(np.diag(D_a))
        D_a_sqrt_inv = inv(D_a_sqrt)
        M = np.dot(np.dot(np.dot(np.dot(D_a_sqrt_inv, inv(S_a)), cov_b), S_a), D_a_sqrt_inv)    # eqn. 10 in Sijs et al.
        D_b, S_b = np.linalg.eigh(M)
        D_gamma = np.diag(np.clip(D_b, a_min=1.0, a_max=None))   # eqn. 11b in Sijs et al.
        return np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(S_a, D_a_sqrt), S_b), D_gamma), inv(S_b)), D_a_sqrt), inv(S_a))  # eqn. 11a in Sijs et al

    
    x_a = x_a.reshape(1, 2)
    x_b = x_b.reshape(1, 2)

    def get_critical_value(dimensions, alpha):
        return chi2.ppf((1 - alpha), df=dimensions)

    eta = get_critical_value(2, 0.05)

    def inv(mat):
        return np.linalg.inv(mat)

    def objective(S):
        return -(S[0]*S[3])

    def constraint1(S):
        S = S.reshape(2, 2).T
        A = inv(C_a) - S@S.T
        return np.linalg.eig(A)[0][0]
    def constraint2(S):
        S = S.reshape(2, 2).T
        A = inv(C_a) - S@S.T
        return np.linalg.eig(A)[0][1]
    def constraint3(S):
        S = S.reshape(2, 2).T
        A = inv(C_b) - S@S.T
        return np.linalg.eig(A)[0][0]
    def constraint4(S):
        S = S.reshape(2, 2).T
        A = inv(C_b) - S@S.T
        return np.linalg.eig(A)[0][1]

    def psuedoinv(A):
        A[np.where(A<=1e-5)] = 1e-5

    def relu(v):
        if v < 100:
            return np.log1p(1 + np.exp(v))
        else:
            return v



    def pinv(A):
        RELU = np.vectorize(relu)
        tmp_eig, tmp_egv = LA.eig(A)
        M_inv = tmp_egv @ np.diag(1/RELU(tmp_eig)) @ tmp_egv.T
        M = tmp_egv @ np.diag(RELU(tmp_eig)) @ tmp_egv.T
        return M, M_inv

    def prob_constraint(S):
        S = S.reshape(2, 2).T
        C_c_inv = S@S.T

        C_ac_inv, C_ac = pinv(inv(C_a) - C_c_inv)
        C_bc_inv, C_bc = pinv(inv(C_b) - C_c_inv)

        _, C_abc_inv_inv = pinv(C_ac_inv + C_bc_inv)
        _, C_abc_inv = pinv(C_ac + C_bc)

        x_c = (C_abc_inv_inv @ (C_ac_inv @ x_a.T + C_bc_inv @ x_b.T)).T
        x_ac = (C_ac @ (inv(C_a) @ x_a.T - C_c_inv @ x_c.T)).T
        x_bc =(C_bc @ (inv(C_b) @ x_b.T - C_c_inv @ x_c.T)).T
        f = ((x_ac - x_bc) @ C_abc_inv @ (x_ac - x_bc).T)[0][0]
        return eta - f

    def constraint5(S):
        return round(S[2], 10)

    con1 = {'type': 'ineq', 'fun': constraint1}
    con2 = {'type': 'ineq', 'fun': constraint2}
    con3 = {'type': 'ineq', 'fun': constraint3}
    con4 = {'type': 'ineq', 'fun': constraint4}
    con5 = {'type': 'ineq', 'fun': prob_constraint}
    con6 = {'type': 'eq', 'fun': constraint5}
    cons = [con1, con2, con3, con4, con5, con6]

    S_0 = 0.9*(np.linalg.cholesky(inv(mutual_covariance(C_a, C_b))).T).reshape(4, )
    prob_constraint(S_0)

    sol = minimize(objective, S_0, method='trust-constr', constraints=cons)
    S = sol.x
    S = S.reshape(2, 2).T

    C_c_05 = inv(S.T) @ inv(S)
    fus_PC_05 = inv(inv(C_a) + inv(C_b) - inv(C_c_05))
    
    x_fus = calculate_MSE(C_a, C_b, C_c_05, x_a, x_b, fus_PC_05)
        
    return fus_PC_05, x_fus

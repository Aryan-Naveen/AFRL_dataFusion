import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2
from scipy.linalg import sqrtm
from numpy.linalg import det
import numpy.linalg as LA
import matplotlib.pyplot as plt
import math
from scipy.stats import multivariate_normal
from scipy.stats import invwishart as iw
from tqdm import tqdm
plt.rc('xtick', labelsize=10) 
plt.rc('ytick', labelsize=10) 
plt.rcParams.update({'font.size': 12})

x_ac_E = []
x_bc_E = []
x_ac_P = []
x_bc_P = []

C_fus_E = []
C_c_E = []
C_fus_P = []
C_c_P = []
C_diff = []
maha = []

C_acs = []
C_bcs = []
C = []

def generate_covariance(true_mu, dims, df):
    S = np.tril(iw.rvs(df, 1, size=dims**2).reshape(dims, dims))
    cov = np.dot(S, S.T)
    while(np.linalg.det(cov) < 1):
        cov = cov * 2
    mu = np.random.multivariate_normal(true_mu, cov, 1)[0]

    return mu, cov

def oneDvisualize(x, C, ax, label, m=1, linestyle = ''):

    b = np.linspace(x - m*C, x+m*C, 1024)
    p = multivariate_normal.pdf(b, mean=x, cov=C)
    if len(linestyle) == 0:
        ax.plot(b, p, label=label)
    else:
        ax.plot(b, p, label=label, linestyle=linestyle)        


    
def get(dims, df):
    true_mu = np.zeros((dims, ))

    x_ac, C_ac = generate_covariance(true_mu, dims, df)
    x_c, C_c = generate_covariance(true_mu, dims, df)
    x_bc, C_bc = generate_covariance(true_mu, dims, df)
    
    C_bc = np.copy(C_ac)
    
    C_a = LA.inv(LA.inv(C_ac) + LA.inv(C_c))
    C_b = LA.inv(LA.inv(C_bc) + LA.inv(C_c))

    x_a = C_a @ (LA.inv(C_ac) @ x_ac + LA.inv(C_c) @ x_c)
    x_b = C_b @ (LA.inv(C_bc) @ x_bc + LA.inv(C_c) @ x_c)

    C_fus = LA.inv(LA.inv(C_a) + LA.inv(C_b) - LA.inv(C_c))

    return x_a.reshape(1, dims), x_b.reshape(1, dims), C_a, C_b, C_fus

C_a_opt = np.linspace(0.01, 2.5, 512)
for index in tqdm(range(512)):
    def get_predef(index):
        x_a = np.array([[1]])
        x_b = np.array([[-1]])
        C_a = np.array([[C_a_opt[index]]])
        C_b = np.array([[1]])
        C_fus = np.array([[1]])
        index += 1
        return x_a, x_b, C_a, C_b, C_fus

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

    dims = 1
    if dims == 2:
        x_a, x_b, C_a, C_b, C_fus = get(2, 6)
    else:
        x_a, x_b, C_a, C_b, C_fus = get_predef(index)
        index += 1
        
    x_a = x_a.reshape(1, dims)
    x_b = x_b.reshape(1, dims)

    def get_critical_value(dimensions, alpha):
        return chi2.ppf((1 - alpha), df=dimensions)



    def inv(mat):
        if dims > 1:
            return np.linalg.inv(mat)
        else:
            return 1/mat

    eta = get_critical_value(dims, 0.01)

    def objective2(S):
        return -(S[0]*S[3])

    def objective1(S):
        return -S[0]

    def constraint1(S):
        S = S.reshape(dims, dims).T
        A = inv(C_a) - S@S.T
        return np.linalg.eig(A)[0][0]
    def constraint2(S):
        S = S.reshape(dims, dims).T
        A = inv(C_a) - S@S.T
        return np.linalg.eig(A)[0][1]
    def constraint3(S):
        S = S.reshape(dims, dims).T
        A = inv(C_b) - S@S.T
        return np.linalg.eig(A)[0][0]
    def constraint4(S):
        S = S.reshape(dims, dims).T
        A = inv(C_b) - S@S.T
        return np.linalg.eig(A)[0][1]
        
    def psuedoinv(A):
        A[np.where(A<=1e-5)] = 1e-5
        
    def relu(v):
        return np.log1p(1 + np.exp(v))
            
    def pinv(A):
        RELU = np.vectorize(relu)
        tmp_eig, tmp_egv = LA.eig(A)
        M_inv = tmp_egv @ np.diag(1/RELU(tmp_eig)) @ tmp_egv.T
        M = tmp_egv @ np.diag(RELU(tmp_eig)) @ tmp_egv.T
        return M, M_inv

    def prob_constraint(S):
        S = S.reshape(dims, dims).T
        C_c_inv = S@S.T

    #     tmp = inv(C_a) - C_c_inv
    #     tmp_eig, tmp_egv = LA.eig(tmp)
    #     C_ac = tmp_egv @ np.diag(1/RELU(tmp_eig)) @ tmp_egv.T
    #     C_ac_inv = tmp_egv @ np.diag(RELU(tmp_eig)) @ tmp_egv.T
        if dims == 2:
            C_ac_inv, C_ac = pinv(inv(C_a) - C_c_inv)
            C_bc_inv, C_bc = pinv(inv(C_b) - C_c_inv)
        
            _, C_abc_inv_inv = pinv(C_ac_inv + C_bc_inv)
            _, C_abc_inv = pinv(C_ac + C_bc)
        elif dims == 1:
            C_ac = inv(inv(C_a) - C_c_inv)
            C_ac_inv = inv(C_ac)
            C_bc = inv(inv(C_b) - C_c_inv)
            C_bc_inv = inv(C_bc)
            
            C_abc_inv_inv = inv(C_ac_inv + C_bc_inv)
            C_abc_inv = inv(C_ac + C_bc)

        
    #     tmp = inv(C_b) - C_c_inv
    #     tmp_eig, tmp_egv = LA.eig(tmp)
    #     C_bc = tmp_egv @ np.diag(1/RELU(tmp_eig)) @ tmp_egv.T
    #     C_bc_inv = tmp_egv @ np.diag(RELU(tmp_eig)) @ tmp_egv.T
            
    #     C_ac = inv(inv(C_a) - C_c_inv)
    #     C_bc = inv(inv(C_b) - C_c_inv)
        x_c = (C_abc_inv_inv @ (C_ac_inv @ x_a.T + C_bc_inv @ x_b.T)).T
        x_ac = (C_ac @ (inv(C_a) @ x_a.T - C_c_inv @ x_c.T)).T
        x_bc =(C_bc @ (inv(C_b) @ x_b.T - C_c_inv @ x_c.T)).T
        f = ((x_ac - x_bc) @ C_abc_inv @ (x_ac - x_bc).T)[0][0]
    #     print(f)
        return eta - f
    def maha2(x_ac, x_bc, C_ac, C_bc):
        return (x_ac - x_bc) * ((C_ac + C_bc)**-1) * (x_ac - x_bc)


    def constraint5(S):
        return S[2]

    if dims == 2:
        con1 = {'type': 'ineq', 'fun': constraint1}
        con2 = {'type': 'ineq', 'fun': constraint2}
        con3 = {'type': 'ineq', 'fun': constraint3}
        con4 = {'type': 'ineq', 'fun': constraint4}
        con5 = {'type': 'eq', 'fun': prob_constraint}
        con6 = {'type': 'eq', 'fun': constraint5}
        cons = [con1, con2, con3, con4, con5, con6]
    if dims == 1:
        con1 = {'type': 'ineq', 'fun': constraint1}
        con3 = {'type': 'ineq', 'fun': constraint3}
        con5 = {'type': 'ineq', 'fun': prob_constraint}
        cons = [con1, con3, con5]

    if dims == 1:
        S_0 = 0.99*(np.linalg.cholesky(inv(mutual_covariance(C_a, C_b))).T).reshape(dims**2, )
    else:
        S_0 = (np.linalg.cholesky(inv(mutual_covariance(C_a, C_b))).T).reshape(dims**2, )
        
    prob_constraint(S_0)
    if dims == 2:
        sol = minimize(objective2, S_0, method='trust-constr', constraints=cons)
    if dims == 1:
        sol = minimize(objective1, S_0, method='SLSQP', constraints=cons)

    def get_x_c(C_c):
        def mutual_mean(mean_a, cov_a, mean_b, cov_b, cov_m):
            dims = mean_a.shape[0]
            cov_m_inv = inv(cov_m)
            cov_a_inv = inv(cov_a)
            cov_b_inv = inv(cov_b)
            H = cov_a_inv + cov_b_inv - np.multiply(2, cov_m_inv)
            if det(H) == 0:
                eta = 0
            else:
                eig_H, _ = np.linalg.eigh(H)
                smallest_nonzero_ev = min(list(filter(lambda x: x != 0, eig_H)))
                eta = 0.0001 * smallest_nonzero_ev
            eta_I = np.multiply(eta, np.identity(dims))
            first_term = inv(cov_a_inv + cov_b_inv - np.multiply(2, cov_m_inv) + np.multiply(2, eta_I))
            second_term = np.dot(cov_b_inv - cov_m_inv + eta_I, mean_a) + np.dot(cov_a_inv - cov_m_inv + eta_I, mean_b)
            return np.dot(first_term, second_term)
        return mutual_mean(x_a, C_a, x_b, C_b, C_c)

    def get_C_ac_x_ac(x_c, C_c):
        C_ac = inv(inv(C_a) - inv(C_c))
        x_ac = C_ac @ (inv(C_a) @ x_a - inv(C_c)@x_c)
        return x_ac, C_ac

    def get_C_bc_x_bc(x_c, C_c):
        C_bc = inv(inv(C_b) - inv(C_c))
        x_bc = C_bc @ (inv(C_b) @ x_b - inv(C_c)@x_c)
        return x_bc, C_bc
    S = sol.x
    S = S.reshape(dims, dims).T

    S = sol.x
    S = S.reshape(dims, dims).T

    C_c_PC = inv(S.T) @ inv(S)
    x_c_PC = get_x_c(C_c_PC)
    C_c_EI = mutual_covariance(C_a, C_b) + 0.01
    x_c_EI = get_x_c(C_c_EI)


    x_acP, C_acP = get_C_ac_x_ac(x_c_PC, C_c_PC)
    x_bcP, C_bcP = get_C_bc_x_bc(x_c_PC, C_c_PC)
    x_ac_P.append(x_acP[0][0])
    x_bc_P.append(x_bcP[0][0])
    fus_PC = inv(inv(C_a) + inv(C_b) - inv(C_c_PC))
    C_fus_P.append(fus_PC[0][0])
    C_c_P.append(C_c_PC[0][0])


    x_acE, C_acE = get_C_ac_x_ac(x_c_EI, C_c_EI)
    x_bcE, C_bcE = get_C_bc_x_bc(x_c_EI, C_c_EI)
    x_ac_E.append(x_acE[0][0])
    x_bc_E.append(x_bcE[0][0])
    fus_EI = inv(inv(C_a) + inv(C_b) - inv(C_c_EI))
    C_fus_E.append(fus_EI[0][0])
    C_c_E.append(C_c_EI[0][0])
    C_diff.append(fus_EI[0][0] - fus_PC[0][0])

    maha.append(maha2(x_acE[0][0], x_bcE[0][0], C_acE[0][0], C_bcE[0][0]))
    C_acs.append(C_acE[0][0])
    C_bcs.append(C_bcE[0][0])
    C.append(C_acE[0][0] + C_bcE[0][0])

w = 1

plt.plot(C_a_opt, x_ac_E, label=r'EI $\mu_{a \backslash c}$', color="red", linewidth=w)
plt.plot(C_a_opt, x_ac_P, label=r'PCEI $\mu_{a \backslash c}$', color="red", linestyle="dashed", linewidth=w)

plt.plot(C_a_opt, x_bc_E, label=r'EI $\mu_{b \backslash c}$', color="blue", linewidth=w)
plt.plot(C_a_opt, x_bc_P, label=r'PCEI $\mu_{a \backslash c}$', color="blue", linestyle="dashed", linewidth=w)

plt.xlabel(r'$C_{a}$ Value')
plt.legend()
plt.show()
plt.savefig("Independent means.png")

plt.cla()
plt.clf()

plt.plot(C_a_opt, C_fus_E, label=r'$C_{EI \phi}$', color="red", linewidth=w)
# plt.plot(C_a_opt, C_c_E, label="Common Covariance EI", color="red", linestyle="dashed", linewidth=w)

plt.plot(C_a_opt, C_fus_P, label=r'$C_{PC \phi}$', color="blue", linewidth=w)
# plt.plot(C_a_opt, C_c_P, label="Common Covariance PC", color="blue", linestyle="dashed", linewidth=w)

plt.plot(C_a_opt, C_diff, label=r'$C_{EI \phi} - C_{PC \phi}$', color="green", linestyle='dashed', linewidth=w)
plt.xlabel(r'$C_{a}$ Value')
plt.legend()
plt.show()

plt.cla()
plt.clf()

plt.plot(C_a_opt, maha, label=r'$\Psi_{EI}$', color="red", linewidth=w)

plt.xlabel(r'$C_{a}$ Value')
plt.legend()
plt.show()

# plt.plot(C_a_opt, C_fus_E, label=r'$C_{EI \phi}$', color="red", linewidth=w)
plt.plot(C_a_opt, C_acs, label="Common Covariance EI", color="red", linestyle="dashed", linewidth=w)

plt.plot(C_a_opt, C_bcs, label=r'$C_{PC \phi}$', color="blue", linewidth=w)
# plt.plot(C_a_opt, C_c_P, label="Common Covariance PC", color="blue", linestyle="dashed", linewidth=w)

plt.plot(C_a_opt, C, label=r'$C_{EI \phi} - C_{PC \phi}$', color="orange", linestyle='dashed', linewidth=w)
plt.xlabel(r'$C_{a}$ Value')
plt.legend()
plt.show()

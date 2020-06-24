import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
from fusionAlgorithms.EllipsoidalKT import EllipsoidalIntersection
from tools.utils import plot_ellipse
import scipy.special as scsp

def z2p(z):
    """From z-score return p-value."""
    return 0.5 * (1 + scsp.erf(z / np.sqrt(2)))


class QCQP_solver():
    def __init__(self, P, q, r, z):
        self.P = P
        self.q = q
        self.r = r
        self.z = z
        self.dims = z.size

    
    def perform_cholesky_transform(self, K):
        S = np.linalg.cholesky(K).T
        self.P = np.linalg.inv(S.T) @ self.P @ np.linalg.inv(S)
        self.q = (self.q.T @ np.linalg.inv(S)).T
        self.z = S @ self.z
    
    def perform_eigen_transform(self):
        e_vals, Q = np.linalg.eig(self.P)
        self.eig = e_vals
        self.P = np.diag(e_vals)
        self.q = Q.T @ self.q
        self.z = Q.T @ self.z
        self.Q = Q
    
    def generate_bounds_for_nu(self):
        bounds = np.zeros((2, 1))
        eig_min = np.min(self.eig)
        eig_max = np.max(self.eig)
        if eig_min>=0:
            print("WARNING:  There is no upper limit for nu. It can go to +\infty")
        if eig_max <= 0:
            print("WARNING:  nu will have to be negative for I+nu \Lambda to be psd. No limits on it")
        bounds[1]= -1/np.min(self.eig)
        bounds[0] = -1/np.max(self.eig)
        print('bounds before sort',bounds,'and after sort',np.sort(bounds))
        self.bounds = np.sort(bounds)

    def calculate_value(self, nu):
        A = self.eig*np.power(nu*self.q-2*self.z, 2)
        B = 4*np.power(1+nu*self.eig, 2)

        C = (self.q@(nu*self.q-2*self.z))
        D = 2*(1+nu*self.eig)
        # print(f'eigs is {self.eig}, q is {self.q}, nu is {nu}, z is {self.z}')
        # print(f'A is {A}, B is {B}, C is {C}, D is {D}, r is {self.r}')
        return np.sum(A/B - C/D) + self.r

    def calculate_derivative(self, nu):
        return -np.sum(np.power(2*self.eig*self.z+self.q, 2)/np.power(2*(1+nu*self.eig), 3))

    def get_potential_nus(self):
        return np.linspace(self.bounds[0]+1E-3, self.bounds[1]-1E-3, abs(int((self.bounds[1]-self.bounds[0])*1024)))    


    def find_optimal_nu(self):
        nus = self.get_potential_nus()
        vfunc = np.vectorize(self.calculate_value)
        dfunc = np.vectorize(self.calculate_derivative)
        
        vals = vfunc(nus)
        derivatives = dfunc(nus)

        return nus[np.argwhere(np.abs(vals) == np.min(np.abs(vals)))][0][0]

    def inverse_eigen_transform(self, x):
        return self.Q @ x
    
    def inverse_cholseky(self, K, x):
        S = np.linalg.cholesky(K).T
        return np.linalg.inv(S) @ x

    def calculate_x_c_val(self, nu):
        return -(0.5)*np.linalg.inv(np.identity(self.dims) + nu * self.P) @ (nu*self.q - 2*self.z)

    def calculate_x_c_case1(self, nu, K):
        x_hat = self.calculate_x_c_val(nu)
        tilde_x_c = self.inverse_eigen_transform(x_hat)
        return self.inverse_cholseky(K, tilde_x_c)

    def case_2(self, nu):
        return (0.5)*np.linalg.pinv(np.identity(self.dims) + nu*self.P) @ (2*self.z - nu*self.q)

    def calculate_x_case2(self, K):
        if self.bounds[0] < 0:
            print("\nLOWER BOUND NU:")
            x_test_hat = self.case_2(self.bounds[0])
            print(x_test_hat.T @ self.P @ x_test_hat + self.q.T @ x_test_hat + self.r)
            print("\n")
        if self.bounds[1] > 0:
            print("\nLOWER BOUND NU:")
            x_test_hat = self.case_2(self.bounds[1])
            print(x_test_hat.T @ self.P @ x_test_hat + self.q.T @ x_test_hat + self.r)
            print("\n")




def calculate_K(C_a, C_b, C_c):

    C_ac_inv = LA.inv(C_a) - LA.inv(C_c)
    C_bc_inv = LA.inv(C_b) - LA.inv(C_c)

    K_a = LA.inv(C_a) @ (LA.inv(C_ac_inv) @ LA.inv(C_a) - np.identity(C_a.shape[0]))
    K_b = LA.inv(C_b) @ (LA.inv(C_bc_inv)@LA.inv(C_b) - np.identity(C_b.shape[0]))
    return K_a, K_b

def calculate_QCQP_Coeff(K_a, K_b, x_a, x_b):
    P = K_a - K_b
    q = -2*(x_a @ K_a - x_b @ K_b)
    r = x_a.T @ K_a @ x_a - x_b.T @ K_b @ x_b
    return P, q, r

def calculate_mahalonobis_difference(x_c, x, K):
    print(x - x_c)
    return (x-x_c) @ K @ (x - x_c).T


def qcqp_solver_x_c(x_a, x_b, C_a, C_b, C_c):
    a = []
    b = []
    multipliers = np.linspace(1, 2, 11)
    g = [95] * multipliers.size
    for m in multipliers:
        C_ac_inv = LA.inv(C_a) - LA.inv(m*C_c)
        C_bc_inv = LA.inv(C_b) - LA.inv(m*C_c)

        K_a, K_b = calculate_K(C_a, C_b, m*C_c)
        P, q, r = calculate_QCQP_Coeff(K_a, K_b, x_a, x_b)

        solver = QCQP_solver(P, q, r, x_a)

        solver.perform_cholesky_transform(K_a)
        solver.perform_eigen_transform()
        solver.generate_bounds_for_nu()

        ################CASE 1#############################
        print("#################MULTIPLIER: " + str(m) + "########################")
        nu = solver.find_optimal_nu()
        x_c = solver.calculate_x_c_case1(nu, K_a)
        a_diff = calculate_mahalonobis_difference(x_c, x_a, K_a)
        b_diff = calculate_mahalonobis_difference(x_c, x_b, K_b)
        print("Mahalonobis difference to A: " + str(a_diff))
        # a.append(z2p(a_diff**0.5)*100)
        print("Mahalonobis difference to B: " + str(b_diff))
        # b.append(z2p(b_diff**0.5)*100)

        print("\n")

    # plt.cla()
    # plt.clf()

    # ax = plt.axes()
    # ax.plot(multipliers, a)
    # ax.plot(multipliers, b)
    # ax.plot(multipliers, g)

    # plt.show()    


    # ################CASE 2#############################
    # nu = solver.find_optimal_nu()
    # x_c = solver.calculate_x_case2(K_a)
    # print("Mahalonobis difference to A: " + str(calculate_mahalonobis_difference(x_c, x_a, K_a)))
    # print("Mahalonobis difference to B: " + str(calculate_mahalonobis_difference(x_c, x_b, K_b)))


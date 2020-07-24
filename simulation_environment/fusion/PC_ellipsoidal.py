from scipy.linalg import sqrtm
from numpy.linalg import det
import numpy.linalg as LA
import torch
from scipy.stats import chi2
import numpy as np
from fusion.gradient_descent import find_direction
import matplotlib.pyplot as plt
import math

from scipy.stats import multivariate_normal
def visualize_one_dimensional(x_a, x_b, x_c, x_d, C_a, C_b, C_c, C_d, filename, multiplier=2):
    lb = min(x_a, x_b) - multiplier*max(abs(C_a[0][0]), abs(C_b[0][0]))
    ub = max(x_a, x_b) + multiplier*max(abs(C_a[0][0]), abs(C_b[0][0]))
    x = np.linspace(lb, ub, 1024, endpoint=False).reshape(1024, )
    ya = multivariate_normal.pdf(x, mean=x_a, cov=C_a[0][0]).reshape(1024, )
    yb = multivariate_normal.pdf(x, mean=x_b, cov=C_b[0][0]).reshape(1024, )
    yc = multivariate_normal.pdf(x, mean=x_c, cov=C_c[0][0]).reshape(1024, )
    yd = multivariate_normal.pdf(x, mean=x_d, cov=C_d[0][0]).reshape(1024, )
    plt.cla()
    plt.clf()
    ax = plt.axes()
    ax.plot(x, ya, label="Independent A distribution")
    ax.plot(x, yb, label="Independent B distribution")
    # ax.plot(x, yc, label="Original A distribution")
    # ax.plot(x, yd, label="Original B distribution")
    ax.legend()
    plt.show()
    plt.savefig(filename)



def get_critical_value(dimensions, alpha):
    return chi2.ppf((1 - alpha), df=dimensions)

def inv(mat):
    return LA.inv(mat)

def MSE(A, B):
    return (np.square(A - B)).mean(axis=None)

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


class DataStorage():
    def __init__(self, C_a, C_b, x_a, x_b):
        self.C_a = C_a
        self.C_b = C_b
        self.C_c = np.zeros(C_a.shape)
        self.ogC_c = np.zeros(C_a.shape)
        self.first = True
        self.x_a = x_a
        self.x_b = x_b
        self.x_c = np.zeros((1, 5))

    def update_x_c(self):
        # C_ac = inv(self.C_a) - inv(self.C_c)
        # C_bc = inv(self.C_b) - inv(self.C_c)
        # self.x_c = (self.C_c @ (C_ac @ self.x_a.T + C_bc @ self.x_b.T)).T
        self.x_c = mutual_mean(self.get_x_a(), self.get_C_a(), self.get_x_b(), self.get_C_b(), self.get_C_c())

    def set_C_c(self, Cc):
        self.C_c = np.copy(Cc)
        self.ogCc = np.copy(Cc)
        self.update_x_c()
    
    def visualize(self, dims, f, m=2):
        if dims == 1:
            C_ac = inv(inv(self.C_a) - inv(self.C_c))
            C_bc = inv(inv(self.C_b) - inv(self.C_c))
            x_ac = (C_ac @ (inv(self.C_a) @ self.x_a.T - inv(self.C_c) @ self.x_c.T)).T
            x_bc = (C_bc @ (inv(self.C_b) @ self.x_b.T - inv(self.C_c) @ self.x_c.T)).T

            visualize_one_dimensional(x_ac, x_bc, self.x_c, self.x_b, C_ac, C_bc, self.C_c, self.C_b, f, multiplier=m)

    def increment_C_c(self, outer, nu):
        self.C_c = self.ogC_c + nu * outer
        self.update_x_c()


    def get_C_a(self, tensor=False):
        if(tensor):
            return torch.tensor(np.copy(self.C_a))
        return np.copy(self.C_a)

    def get_C_b(self, tensor=False):
        if(tensor):
            return torch.tensor(np.copy(self.C_b))
        return np.copy(self.C_b)

    def get_C_c(self, tensor=False):
        if(tensor):
            return torch.tensor(np.copy(self.C_c))
        return np.copy(self.C_c)

    def get_x_a(self, tensor=False):
        if(tensor):
            return torch.tensor(np.copy(self.x_a))
        return np.copy(self.x_a)

    def get_x_b(self, tensor=False):
        if(tensor):
            return torch.tensor(np.copy(self.x_b))
        return np.copy(self.x_b)
    
    def calculate_mahalanobis_distance(self):
        C_ac = inv(inv(self.C_a) - inv(self.C_c))
        C_bc = inv(inv(self.C_b) - inv(self.C_c))
        x_ac = (C_ac @ (inv(self.C_a) @ self.x_a.T - inv(self.C_c) @ self.x_c.T)).T
        x_bc = (C_bc @ (inv(self.C_b) @ self.x_b.T - inv(self.C_c) @ self.x_c.T)).T
        return abs((x_ac - x_bc) @ inv(C_ac + C_bc) @ (x_ac - x_bc).T)
    
    def calculate_fusion_covariance(self):
        C_ac = inv(inv(self.C_a) - inv(self.C_c))
        C_bc = inv(inv(self.C_b) - inv(self.C_c))
        return inv(inv(self.C_a) + inv(self.C_b) - inv(self.C_c))

    def calculate_fusion_mean(self):
        C_fus = self.calculate_fusion_covariance()
        C_ac = inv(inv(self.C_a) - inv(self.C_c))
        C_bc = inv(inv(self.C_b) - inv(self.C_c))
        x_ac = (C_ac @ (inv(self.C_a) @ self.x_a.T - inv(self.C_c) @ self.x_c.T)).T
        x_bc = (C_bc @ (inv(self.C_b) @ self.x_b.T - inv(self.C_c) @ self.x_c.T)).T
        return C_fus @ (LA.inv(C_ac) @ x_ac + LA.inv(C_bc) @ x_bc + LA.inv(self.C_c) @ self.x_c)


def mutual_covariance(cov_a, cov_b):
    D_a, S_a = np.linalg.eigh(cov_a)
    D_a_sqrt = sqrtm(np.diag(D_a))
    D_a_sqrt_inv = inv(D_a_sqrt)
    M = np.dot(np.dot(np.dot(np.dot(D_a_sqrt_inv, inv(S_a)), cov_b), S_a), D_a_sqrt_inv)    # eqn. 10 in Sijs et al.
    D_b, S_b = np.linalg.eigh(M)
    D_gamma = np.diag(np.clip(D_b, a_min=1.0, a_max=None))   # eqn. 11b in Sijs et al.
    return np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(S_a, D_a_sqrt), S_b), D_gamma), inv(S_b)), D_a_sqrt), inv(S_a))  # eqn. 11a in Sijs et al

def avg(arr):
    return sum(arr)/len(arr)

def binary_search(data, z, cr):
    ogC_c = data.get_C_c()
    outer = z.T @ z
    bounds = 1e-30
    eta = 1e-2
    i = 0
    while(True):
        i += 1
        data.set_C_c(ogC_c + bounds*outer)
        val = data.calculate_mahalanobis_distance()
        if(val < cr) or (i > 10000):
            break
        bounds += eta

def mahalanobis_dist_to_true(x_true, data):
    x_c = data.calculate_fusion_mean()
    C_true = data.calculate_fusion_covariance()
    return ((x_true - x_c).T @ inv(C_true) @ (x_true - x_c))**2


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


def fusion(C_a, C_b, x_a, x_b, true_Cc, true_xc, C_c):
    plt.cla()
    plt.clf()
    ax = plt.axes()
    plot_ellipse(true_Cc, ax, "true fusion")
    # plot_ellipse(C_a, ax)
    # plot_ellipse(C_b, ax, "initial distribution")
    # plot_ellipse(C_c, ax, "true common info", alpha_val=0.1)


    dims = x_a.size
    data = DataStorage(C_a, C_b, x_a, x_b)
    data.set_C_c(mutual_covariance(C_a, C_b))

    plot_ellipse(data.get_C_c(), ax, label_t="C_c EI", alpha_val=0.75, linestyle='dashed', color_def="blue")
    plot_ellipse(data.calculate_fusion_covariance(), ax, label_t="C_fus EI", alpha_val=1, color_def="blue")


    a = MSE(data.calculate_fusion_mean(), true_xc) + MSE(data.calculate_fusion_covariance(), true_Cc)

    det1 = LA.det(data.calculate_fusion_covariance())
    dist1 = mahalanobis_dist_to_true(true_xc, data)
    z = find_direction(data)

    cr_05 = get_critical_value(dims, 0.01)
    if data.calculate_mahalanobis_distance() > cr_05:
        binary_search(data, z, cr_05)

    # plot_ellipse(data.get_C_c(), ax, label_t="C_c PC", alpha_val=0.75, linestyle='dashed', color_def="green")
    plot_ellipse(data.calculate_fusion_covariance(), ax, label_t="C_fus PC", alpha_val=1, color_def="green")

    data.visualize(dims, "pcei.png")
    b = MSE(data.calculate_fusion_mean(), true_xc) + MSE(data.calculate_fusion_covariance(), true_Cc)
    det2 = LA.det(data.calculate_fusion_covariance())
    dist2 = mahalanobis_dist_to_true(true_xc, data)
    ax.legend()
    plt.show()

    smaller_det = np.linalg.det(true_Cc) > np.linalg.det(data.calculate_fusion_covariance())
    larger_det = np.linalg.det(C_c) < np.linalg.det(data.get_C_c())

    return a, det1, dist1, b, det2, dist2, smaller_det, larger_det
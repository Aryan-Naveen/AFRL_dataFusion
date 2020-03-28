import numpy as np
import numpy.linalg as lin
from KL_divergence import compute_KL
import math
import random
from scipy.stats import multivariate_normal
from tqdm import tqdm
import matplotlib.pyplot as plt

def plot_ellipse(covariance, ax, label_t):
    x_el = np.array([np.sin(np.linspace(0, 2*math.pi, num=63)), np.cos(np.linspace(0, 2*math.pi, num=63))])
    C = np.linalg.cholesky(covariance)
    y_el = np.dot(C, x_el)
    ax.plot(y_el[0], y_el[1], label=label_t)


def CI(mu_s, cov_s):
    w = 1./len(mu_s)
    total_cov = np.zeros(cov_s[0].shape)
    total_mu = np.zeros(mu_s[0].shape)
    for mu, cov in zip(mu_s, cov_s):
        total_mu = np.add(total_mu, w*np.dot(lin.inv(cov), mu))
        total_cov = np.add(total_cov, w*lin.inv(cov))
    
    comb_cov = lin.inv(total_cov)
    mu_comb = np.dot(comb_cov, total_mu)
    return mu_comb, comb_cov

def CI_update(mu_a, cov_a, mu_b, cov_b, w):
    cov_comb = lin.inv(np.add(w*np.linalg.inv(cov_a), (1-w)*lin.inv(cov_b)))
    sens_1 = w*np.dot(np.linalg.inv(cov_a), mu_a)
    sens_2 = (1-w)*np.dot(np.linalg.inv(cov_b), mu_b)
    mu_comb = np.dot(cov_comb, (np.add(sens_1, sens_2)))
    return mu_comb, cov_comb

def get_weights(sensor_covariances):
    determinats = np.array([np.linalg.det(cov) for cov in sensor_covariances])
    c = np.copy(determinats)
    m_det = max(determinats)
    determinats = np.array([m_det - det for det in determinats])
    weights = determinats/np.linalg.norm(determinats)
    return weights, c

def covarianceIntersection(sensor_readings, sensor_covariances, N_time_steps, neighbors, N_agents, KL_inp, true_dist):
    KL_div = {}
    ax = plt.axes()
    determinant_progression = {}
    for i in range(N_agents):
        KL_div[i] = []
        determinant_progression[i] = [lin.det(sensor_covariances[i])]
    print("Covariance Intersection...")
    for i in tqdm(range(N_time_steps)):
        for j in range(N_agents):
            mu_s = [sensor_readings[j]]
            cov_s = [sensor_covariances[j]]
            if i==0:
                plot_ellipse(sensor_covariances[j], ax, "Sensor " + str(j))
            for k in neighbors[j][0]:
                mu_s.append(sensor_readings[k])
                cov_s.append(sensor_covariances[k])
            sensor_readings[j], sensor_covariances[j] = CI(mu_s, cov_s)
            dist = multivariate_normal(sensor_readings[j], sensor_covariances[j])
            KL_div[j].append(compute_KL(true_dist, dist, KL_inp))
            determinant_progression[j].append(lin.det(sensor_covariances[j]))
    plot_ellipse(sensor_covariances[0], ax, "FINAL")
    plt.legend(loc='upper left', borderaxespad=0.)
    plt.grid(b = True)
    plt.title("Covariance Ellipses")
    plt.show()
    return sensor_readings, sensor_covariances, KL_div, determinant_progression

import numpy as np
import numpy.linalg as lin
from KL_divergence import compute_KL
import math
import random
from scipy.stats import multivariate_normal
from tqdm import tqdm
import matplotlib.pyplot as plt

import cProfile, pstats, io
from pstats import SortKey

def plot_ellipse(covariance, ax, label_t):
    x_el = np.array([np.sin(np.linspace(0, 2*math.pi, num=63)), np.cos(np.linspace(0, 2*math.pi, num=63))])
    C = np.linalg.cholesky(covariance)
    y_el = np.dot(C, x_el)
    ax.plot(y_el[0], y_el[1], label=label_t)

def plot_last_ellipse(covariance, ax, label_t):
    x_el = np.array([np.sin(np.linspace(0, 2*math.pi, num=63)), np.cos(np.linspace(0, 2*math.pi, num=63))])
    C = np.linalg.cholesky(covariance)
    y_el = np.dot(C, x_el)
    ax.plot(y_el[0], y_el[1], label=label_t, linewidth=3)

def CI(mu_s, cov_s, weights):
    total_cov = np.zeros(cov_s[0].shape)
    total_mu = np.zeros(mu_s[0].shape)
    for mu, cov, w in zip(mu_s, cov_s, weights):
        total_mu = np.add(total_mu, w*np.dot(lin.inv(cov), mu))
        total_cov = np.add(total_cov, w*lin.inv(cov))
    
    comb_cov = lin.inv(total_cov)
    mu_comb = np.dot(comb_cov, total_mu)
    return mu_comb, comb_cov

def find_optimal_cov_and_mu(iter, mu_s, cov_s):
    N_agents = len(mu_s)
    default_weights = np.full((N_agents, ), 1./len(mu_s))
    mu, cov = CI(mu_s, cov_s, default_weights)
    for i in range(iter):
        weights = np.random.rand(N_agents, )
        weights/= sum(weights)
        pot_mu, pot_cov = CI(mu_s, cov_s, weights)
        if(lin.det(pot_cov) < lin.det(cov)):
            mu, cov = pot_mu, pot_cov
    return mu, cov

def get_weights(sensor_covariances):
    determinats = np.array([np.linalg.det(cov) for cov in sensor_covariances])
    c = np.copy(determinats)
    m_det = max(determinats)
    determinats = np.array([m_det - det for det in determinats])
    weights = determinats/np.linalg.norm(determinats)
    return weights, c

def covarianceIntersection(sensor_readings, sensor_covariances, N_time_steps, neighbors, N_agents, KL_inp, true_dist, calculate_KL=True, calculate_det=True):
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
            sensor_readings[j], sensor_covariances[j] = find_optimal_cov_and_mu(100, mu_s, cov_s)
            if(calculate_KL):
                dist = multivariate_normal(sensor_readings[j], sensor_covariances[j])
                KL_div[j].append(compute_KL(true_dist, dist, KL_inp))
            if(calculate_det):
                determinant_progression[j].append(lin.det(sensor_covariances[j]))
    plot_last_ellipse(sensor_covariances[0], ax, "FINAL")
    plt.legend(loc='upper left', borderaxespad=0.)
    plt.grid(b = True)
    plt.title("Covariance Ellipses")
    plt.savefig("Covariance Ellipses.png")
    plt.show()
    return sensor_readings, sensor_covariances, KL_div, determinant_progression

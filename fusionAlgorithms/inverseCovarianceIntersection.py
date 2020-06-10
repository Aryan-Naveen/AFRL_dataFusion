
from tools.KL_divergence import compute_KL
from tools.utils import check_if_singular

import numpy as np
import numpy.linalg as LA
import math
import random
from scipy.stats import multivariate_normal
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm

def CI(mu_s, cov_s, weights):
    total_cov = np.zeros(cov_s[0].shape)
    for mu, cov, w in zip(mu_s, cov_s, weights):
        total_cov = np.add(total_cov, w*LA.inv(cov))

    total_cov = LA.inv(total_cov)
    return total_cov

def ICI(mu_a, cov_a, mu_b, cov_b, w_1, w_2):
    C_cov = w_1*cov_a + w_2*cov_b
    C_ICI = LA.inv(LA.inv(cov_a) + LA.inv(cov_b) - LA.inv(C_cov))
    K = C_ICI@(LA.inv(cov_a) - w_1*LA.inv(w_1*cov_a + w_2*cov_b))
    L = C_ICI@(LA.inv(cov_b) - w_2*LA.inv(w_1*cov_a + w_2*cov_b))
    x_ICI = K @ mu_a + L @ mu_b
    return x_ICI, C_ICI, C_cov

    

def find_optimal_cov_and_mu_random_sampling(iterations, mu_s, cov_s, count):
    N_agents = len(mu_s)
    default_weights = np.full((N_agents, ), 1./len(mu_s))
    mu, cov = ICI(mu_s, cov_s, default_weights)
    count = 0
    count_used = False
    for i in range(iterations):
        weights = np.random.rand(N_agents, )
        while(not sum(weights) == 1):
            weights = np.random.rand(N_agents, )
            weights /= sum(weights)
        pot_mu, pot_cov = ICI(mu_s, cov_s, weights)
        count_used = False
        if LA.det(pot_cov) < LA.det(cov):
            count_used = True
            mu, cov = pot_mu, pot_cov
    if(count_used):
        count += 1.0
    return mu, cov, count

def find_optimal_cov_and_mu_analytical(iter, mu_s, cov_s, count):
    N_agents = len(mu_s)
    default_weights = np.full((N_agents, ), 1./len(mu_s))
    mu, cov = ICI(mu_s, cov_s, default_weights)
    det = np.array([LA.det(c) for c in cov_s])
    weights = det/np.sum(det)
    weights = max(weights) - weights
    if(sum(weights) == 0):
        weights = default_weights
    pot_mu, pot_cov = ICI(mu_s, cov_s, weights)
    count_used = False
    if LA.det(pot_cov) < LA.det(cov):
        count_used = True
        mu, cov = pot_mu, pot_cov
    if(count_used):
        count += 1.0
    return mu, cov, count


def inverseCovarianceIntersection(sensor_readings, sensor_covariances, N_time_steps, neighbors, N_agents, KL_inp, true_dist, calculate_KL=False, calculate_det=True):
    count = 0.0
    KL_div = {}
    determinant_progression = {}
    next_data = []
    for i in range(N_agents):
        next_data.append([[], []])
        KL_div[i] = []
        determinant_progression[i] = [LA.det(sensor_covariances[i])]
    print("Covariance Intersection...")
    for i in tqdm(range(N_time_steps)):
        for j in range(N_agents):
            mu_s = [sensor_readings[j]]
            cov_s = [sensor_covariances[j]]
            for k in neighbors[j][0]:
                mu_s.append(sensor_readings[k])
                cov_s.append(sensor_covariances[k])
            next_data[j][0], next_data[j][1], count = find_optimal_cov_and_mu_random_sampling(0, mu_s, cov_s, count)
            if(calculate_KL):
                dist = multivariate_normal(next_data[j][0], next_data[j][1])
                KL_div[j].append(compute_KL(true_dist, dist, KL_inp))
            if(calculate_det):
                determinant_progression[j].append(LA.det(next_data[j][1]))
        for j in range(N_agents):
            sensor_readings[j] = next_data[j][0]
            sensor_covariances[j] = next_data[j][1]

    return sensor_readings, sensor_covariances, KL_div, determinant_progression


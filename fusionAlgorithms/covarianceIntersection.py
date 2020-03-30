
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
    total_mu = np.zeros(mu_s[0].shape)
    for mu, cov, w in zip(mu_s, cov_s, weights):
        total_mu = np.add(total_mu, w*np.dot(LA.inv(cov), mu))
        total_cov = np.add(total_cov, w*LA.inv(cov))
    
    comb_cov = LA.inv(total_cov)
    mu_comb = np.dot(comb_cov, total_mu)
    return mu_comb, comb_cov

def find_optimal_cov_and_mu_random_sampling(iter, mu_s, cov_s, count):
    N_agents = len(mu_s)
    default_weights = np.full((N_agents, ), 1./len(mu_s))
    mu, cov = CI(mu_s, cov_s, default_weights)
    for i in range(iter):
        weights = np.random.rand(N_agents, )
        weights/= sum(weights)
        pot_mu, pot_cov = CI(mu_s, cov_s, weights)
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
    mu, cov = CI(mu_s, cov_s, default_weights)
    det = np.array([LA.det(c) for c in cov_s])
    weights = det/np.sum(det)
    weights = max(weights) - weights
    pot_mu, pot_cov = CI(mu_s, cov_s, weights)
    count_used = False
    if LA.det(pot_cov) < LA.det(cov):
        count_used = True
        mu, cov = pot_mu, pot_cov
    if(count_used):
        count += 1.0
    return mu, cov, count


def covarianceIntersection(sensor_readings, sensor_covariances, N_time_steps, neighbors, N_agents, KL_inp, true_dist, calculate_KL=False, calculate_det=True):
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
            next_data[j][0], next_data[j][1], count = find_optimal_cov_and_mu_analytical(100, mu_s, cov_s, count)
            if(calculate_KL):
                dist = multivariate_normal(sensor_readings[j], sensor_covariances[j])
                KL_div[j].append(compute_KL(true_dist, dist, KL_inp))
            if(calculate_det):
                determinant_progression[j].append(LA.det(sensor_covariances[j]))
        for j in range(N_agents):
            sensor_readings[j] = next_data[j][0]
            sensor_covariances[j] = next_data[j][1]
        converged = True
        ref_det = round(LA.det(sensor_covariances[0]), 10)
        for k in range(1, N_agents):
            if not round(LA.det(sensor_covariances[k]), 10) == ref_det:
                converged = False
                break
        if converged:
            N_time_steps = i
            break

    percent = (count/(N_time_steps*N_agents))*100
    print("Sensors converged after " + str(i) + " time steps...")
    print("Percent of time alternative weights used: " + str(percent) + " %")
    return sensor_readings, sensor_covariances, KL_div, determinant_progression


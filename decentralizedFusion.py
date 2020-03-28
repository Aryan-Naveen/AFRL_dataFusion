import numpy as np
import numpy.linalg as lin
from KL_divergence import compute_KL
import math
import random
from scipy.stats import multivariate_normal
from tqdm import tqdm

def CI_update(mu_a, cov_a, mu_b, cov_b, w):
    cov_comb = lin.inv(np.add(w*np.linalg.inv(cov_a), (1-w)*lin.inv(cov_b)))
    sens_1 = w*np.dot(np.linalg.inv(cov_a), mu_a)
    sens_2 = (1-w)*np.dot(np.linalg.inv(cov_b), mu_b)
    mu_comb = np.dot(cov_comb, (np.add(sens_1, sens_2)))
    return mu_comb, cov_comb

def covarianceIntersection(sensor_readings, sensor_covariances, N_time_steps, neighbors, N_agents, KL_inp, true_dist):
    KL_div = {}
    for i in range(N_agents):
        KL_div[i] = []
    print("Covariance Intersection...")
    for i in tqdm(range(N_time_steps)):
        for j in range(N_agents):
            sensor_mu = sensor_readings[j]
            sensor_cov = sensor_covariances[j]
            for k in neighbors[j][0]:
                mu_k = sensor_readings[k]
                cov_k = sensor_covariances[k]
                sensor_mu, sensor_cov = CI_update(sensor_mu, sensor_cov, mu_k, cov_k, 0.5)

            sensor_readings[j] = sensor_mu
            sensor_covariances[j] = sensor_cov
            dist = multivariate_normal(sensor_mu, sensor_cov)
            KL_div[j].append(compute_KL(true_dist, dist, KL_inp))

    return sensor_readings, sensor_covariances, KL_div
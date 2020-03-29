import numpy as np
import numpy.linalg as LA
from KL_divergence import compute_KL
import math
import random
from scipy.stats import multivariate_normal
from tqdm import tqdm
import matplotlib.pyplot as plt

import cProfile, pstats, io
from pstats import SortKey
from numpy import dot


def plot_last_ellipse(covariance, ax, label_t):
    x_el = np.array([np.sin(np.linspace(0, 2*math.pi, num=63)), np.cos(np.linspace(0, 2*math.pi, num=63))])
    C = np.linalg.cholesky(covariance)
    y_el = np.dot(C, x_el)
    ax.plot(y_el[0], y_el[1], label=label_t, linewidth=3)

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


def get_weights(sensor_covariances):
    determinats = np.array([np.linalg.det(cov) for cov in sensor_covariances])
    c = np.copy(determinats)
    m_det = max(determinats)
    determinats = np.array([m_det - det for det in determinats])
    weights = determinats/np.linalg.norm(determinats)
    return weights, c

def covarianceIntersection(sensor_readings, sensor_covariances, N_time_steps, neighbors, N_agents, KL_inp, true_dist, calculate_KL=False, calculate_det=True):
    count = 0.0
    KL_div = {}
    determinant_progression = {}
    for i in range(N_agents):
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
            sensor_readings[j], sensor_covariances[j], count = find_optimal_cov_and_mu_analytical(100, mu_s, cov_s, count)
            if(calculate_KL):
                dist = multivariate_normal(sensor_readings[j], sensor_covariances[j])
                KL_div[j].append(compute_KL(true_dist, dist, KL_inp))
            if(calculate_det):
                determinant_progression[j].append(LA.det(sensor_covariances[j]))
        converged = True
        ref_det = round(LA.det(sensor_covariances[0]), 10)
        for k in range(1, N_agents):
            if not round(LA.det(sensor_covariances[k]), 10) == ref_det:
                converged = False
                break
        if converged:
            break

    percent = (count/(N_time_steps*N_agents))*100
    print("Percent of time alternative weights used: " + str(percent) + " %")
    return sensor_readings, sensor_covariances, KL_div, determinant_progression

def MutualCovariance(cov_i, cov_j):
    _, S_i = LA.eig(cov_i)
    D_i = np.diag(LA.eigvals(cov_i))
    D_i_05 = D_i ** 0.5
    D_i_neg = np.where(D_i>1e-320, D_i**-0.5, 0)
    comb = D_i_neg @ LA.inv(S_i) @ cov_j @ S_i @ D_i_neg
    _, S_j = LA.eig(comb)
    D_j = np.diag(LA.eigvals(comb))
    D_T = np.clip(np.diag(np.diagonal(D_j)), 0, 1)
    S_i_inv = LA.inv(S_i)
    S_j_inv = LA.inv(S_j)
    return S_i @ D_i_05 @ S_j @ D_T @ S_j_inv @ D_i_05 @ S_i_inv

def check_if_singular(M):
    return LA.det(M) == 0

def MutualMean(cov_i, cov_j, mut_cov, mu_i, mu_j):
    if(check_if_singular(cov_i)):
        return [], True
    if(check_if_singular(cov_j)):
        return [], True
    if(check_if_singular(mut_cov)):
        return [], True
    H = LA.inv(cov_i) + LA.inv(cov_j) - 2*LA.inv(mut_cov)
    lamda = 0
    if round(LA.det(H), 50) == 0:
        (w, v) = LA.eigvals(H)
        w = np.where(w!=0, w, np.inf)
        lamda = np.amin(np.where(w != 0,w, np.inf))

    inside = LA.inv(cov_i) + LA.inv(cov_j) - 2*LA.inv(mut_cov) +2*lamda
    if(check_if_singular(inside)):
        return [], True
    mat_1 = LA.inv(inside)
    mat_2 = np.dot(LA.inv(cov_j) - LA.inv(mut_cov) + lamda, mu_i)
    mat_3 = np.dot(LA.inv(cov_i) - LA.inv(mut_cov) + lamda, mu_j)
    return  np.dot(mat_1, (mat_2 + mat_3)), False

def ellipsoidalIntersection(sensor_mus, sensor_covariarances, neighbors, time_steps=1000, track_KL=False, track_det=True, true_dist=0, KL_inp=[]):
    N_agents = len(sensor_mus)
    edge_list = []
    KL_div = {}
    determinant_progression = {}
    for i in range(N_agents):
        KL_div[i] = []
        determinant_progression[i] = [LA.det(sensor_covariarances[i])]
        for num in neighbors[i][0]:
            if not [num, i] in edge_list:
                edge_list.append([i, num])
    print("Ellipsoidal Intersection...")
    for t in tqdm(range(time_steps)):
        for edge in edge_list:
            i = edge[0]
            k = edge[1]        
            sensor_mu = sensor_mus[i]
            sensor_cov = sensor_covariarances[i]
            neighbor_mu = sensor_mus[k]
            neighbor_cov = sensor_covariarances[k]
            mut_cov = MutualCovariance(sensor_cov, neighbor_cov)
            mut_mu, converged = MutualMean(sensor_cov, neighbor_cov, mut_cov, sensor_mu, neighbor_mu)
            if(converged):
                break
            sensor_mus[i], sensor_mus[k] = mut_mu, mut_mu
            sensor_covariarances[i], sensor_covariarances[k] = mut_cov, mut_cov
        if(converged):
            break
        if(track_KL):
            for j in range(N_agents):
                dist = multivariate_normal(sensor_mus[j], sensor_covariarances[j])
                KL_div[j].append(compute_KL(true_dist, dist, KL_inp))
        if(track_det):
            for j in range(N_agents):
                determinant_progression[j].append(LA.det(sensor_covariarances[j]))
    return sensor_mus, sensor_covariarances, KL_div, determinant_progression

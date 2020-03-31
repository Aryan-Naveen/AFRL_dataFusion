import numpy as np
import numpy.linalg as LA
from scipy.linalg import sqrtm
from tqdm import tqdm

def eigen_decompostiion(A):
    #A=S*D*S^-1
    d, S = LA.eigh(A)
    D = np.diag(d)
    return S, D


def EI_fusion(P_i_orig, P_j_orig, mu_i, mu_j):
    #Find mutual covariance and mutual mean
    #For mutual covariance perform eigen decomposition on P_i
    P_i = np.copy(P_i_orig)
    P_j = np.copy(P_j_orig)
    dims = P_i.shape[0]
    S_i, D_i = eigen_decompostiion(P_i)
    P_prime_j = LA.inv(D_i)**0.5 @ LA.inv(S_i) @ P_j @ S_i @ (LA.inv(D_i)**0.5)
    S_j, D_j = eigen_decompostiion(P_prime_j)
    D_T = np.diag([min(val, 1) for val in np.diagonal(D_j)])
    P_mut = S_i @ D_i**0.5 @ S_j @ D_T @ LA.inv(S_j) @ D_i**0.5 @ LA.inv(S_i)
    #Mutual Mean formula 
    #W_i = P_i^-1 - P_mut^-1 
    #W_j = P_j^-1 - P_mut^-1 
    #mut_mu = (W_i + W_j)^-1 * (W_i*mu_i + W_j * mu_j)
    #Cannot guarantee W_i and W_j are positive definite (Need to add slight identity matrix)    
    H = LA.inv(P_i) + LA.inv(P_j)-2*LA.inv(P_mut)
    #H is a fishers matrix and represents input information (if 0 there is an issue)
    eta = 0
    if LA.det(H) == 0:
        e_vals, _ = LA.eigh(H)
        e_vals = np.where(e_vals==0, np.inf, e_vals)
        eta = np.amin(e_vals)
    if eta == np.inf:
        print(eta)
    eta_i = np.multiply(eta,np.identity(dims))
    W_i = LA.inv(P_i) - LA.inv(P_mut) + eta_i
    W_j = LA.inv(P_j) - LA.inv(P_mut) + eta_i
    mu_mut = np.dot(LA.inv(np.add(W_i, W_j).real), np.dot(W_i, mu_i) + np.dot(W_j, mu_j))
    P_fused = LA.inv(LA.inv(P_i.real) + LA.inv(P_j.real) - LA.inv(P_mut.real))
    mu_fused = np.dot(P_fused, np.dot(LA.inv(P_i), mu_i) + np.dot(LA.inv(P_j), mu_j) - np.dot(LA.inv(P_mut), mu_mut))
    return mu_fused, P_fused


def ellipsoidalIntersection(sensor_readings, sensor_covariances, N_time_steps, neighbors, N_agents, KL_inp, true_dist, calculate_KL=False, calculate_det=True):
    count = 0.0
    KL_div = {}
    determinant_progression = {}
    next_data = []
    for i in range(N_agents):
        next_data.append([[], []])
        KL_div[i] = []
        determinant_progression[i] = [LA.det(sensor_covariances[i])]
    print("Ellipsoidal Intersection...")
    for i in tqdm(range(N_time_steps)):
        for j in range(N_agents):
            s_mu = sensor_readings[j]
            s_cov = sensor_covariances[j]
            for k in neighbors[j][0]:
                s_mu, s_Cov = EI_fusion(s_cov, sensor_covariances[k], s_mu, sensor_readings[k])
            next_data[j] = [s_mu, s_cov]
            if(calculate_KL):
                dist = multivariate_normal(sensor_readings[j], sensor_covariances[j])
                KL_div[j].append(compute_KL(true_dist, dist, KL_inp))
            if(calculate_det):
                determinant_progression[j].append(LA.det(sensor_covariances[j]))
        for k in range(N_agents):
            sensor_readings[k], sensor_covariances[k] = next_data[k][0], next_data[k][1] 

    print("Sensors converged after " + str(i) + " time steps...")
    return sensor_readings, sensor_covariances, KL_div, determinant_progression
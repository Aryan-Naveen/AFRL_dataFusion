import numpy as np
import numpy.linalg as LA
from scipy.linalg import sqrtm
from scipy.linalg import svd
from tqdm import tqdm


def eigen_decompostiion(A):
    U, s, _ = svd(A)
    return U, np.diag(s)


def EI_fusion(P_i_orig, P_j_orig, mu_i, mu_j):
    P_i = np.copy(P_i_orig)
    P_i_inv = LA.inv(P_i)
    P_j = np.copy(P_j_orig)
    P_j_inv = LA.inv(P_j)
    dims = P_i_orig.shape[0]
    #Find mutual covariance and mutual mean
    #For mutual covariance perform eigen decomposition on P_i
    S_i, D_i = eigen_decompostiion(P_i)
    D_i_sqrt = sqrtm(D_i)
    D_i_inv_sqrt = LA.inv(D_i_sqrt)
    P_prime_j =  np.dot(np.dot(np.dot(np.dot(D_i_inv_sqrt, LA.inv(S_i)), P_j), S_i), D_i_inv_sqrt) 
    S_j, D_j = eigen_decompostiion(P_prime_j)
    D_T = np.diag([max(val, 1) for val in np.diagonal(D_j)])
    P_mut = np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(S_i, D_i_sqrt), S_j), D_T), LA.inv(S_j)), D_i_sqrt), LA.inv(S_i))
    #Mutual Mean formula 
    #W_i = P_i^-1 - P_mut^-1 
    #W_j = P_j^-1 - P_mut^-1 
    #mut_mu = (W_i + W_j)^-1 * (W_i*mu_i + W_j * mu_j)
    #Cannot guarantee W_i and W_j are positive definite (Need to add slight identity matrix)    
    H = LA.inv(P_i) + LA.inv(P_j)- np.multiply(2, LA.inv(P_mut))
    #H is a fishers matrix and represents input information (if 0 there is an issue)
    P_mut_inv = LA.inv(P_mut)
    eta = 0
    if LA.det(H) == 0:
        eig_H, _ = np.linalg.eigh(H)
        smallest_nonzero_ev = min(list(filter(lambda x: x != 0, eig_H)))
        eta = 0.0001 * smallest_nonzero_ev
    eta_I = np.multiply(eta,np.identity(dims))
    first_term = LA.inv(P_i_inv + P_j_inv - np.multiply(2, P_mut_inv) + np.multiply(2, eta_I))
    second_term = np.dot(P_j_inv - P_mut_inv + eta_I, mu_i) + np.dot(P_i_inv - P_mut_inv + eta_I, mu_j)    
    mu_mut = np.dot(first_term, second_term)
    # mu_mut = np.dot(LA.inv(np.add(W_i, W_j)), np.dot(W_i, mu_i) + np.dot(W_j, mu_j))
    P_fused = LA.inv(LA.inv(P_i) + LA.inv(P_j) - LA.inv(P_mut))
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
            s_mu = np.copy(sensor_readings[j])
            s_cov = np.copy(sensor_covariances[j])
            for k in neighbors[j][0]:
                s_mu, s_cov = EI_fusion(s_cov, sensor_covariances[k], s_mu, sensor_readings[k])
            next_data[j] = [s_mu, s_cov]
            if(calculate_KL):
                dist = multivariate_normal(next_data[j][0], next_data[j][1])
                KL_div[j].append(compute_KL(true_dist, dist, KL_inp))
            if(calculate_det):
                determinant_progression[j].append(LA.det(next_data[j][1]))
        for k in range(N_agents):
            sensor_readings[k], sensor_covariances[k] = next_data[k][0], next_data[k][1] 

    print("Sensors converged after " + str(i) + " time steps...")
    return sensor_readings, sensor_covariances, KL_div, determinant_progression
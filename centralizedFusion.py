import numpy as np
from scipy.stats import norm
import random

def centralizedAlgorithm(sensor_covs, sensor_readings):
    I_c = np.zeros(sensor_covs[0].shape)
    i_c = np.zeros(sensor_readings[0].shape)
    for s_cov, s_mu in zip(sensor_covs, sensor_readings):
        s_cov_inv = np.linalg.inv(s_cov)
        I_c += s_cov_inv
        i_c += np.dot(s_cov_inv, s_mu)
    
    fused_cov = np.linalg.inv(I_c)
    fused_mu = np.dot(fused_cov, i_c)

    return fused_cov, fused_mu
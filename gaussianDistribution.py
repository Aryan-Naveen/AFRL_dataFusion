import numpy as np
from utils import toTuple

def generate_samples(dims, num_samples, num_sensors, target_loc, sens):
    true_samples = np.ndarray(shape=(num_sensors, num_samples, dims))
    covs = []
    for i in range(num_sensors):
        sensor_loc = sens[:, i]
        mu = toTuple(sensor_loc - target_loc)
        sigma = toTuple(np.random.rand(dims, ))
        true_samples[i] = np.random.normal(mu, sigma, (num_samples, dims))
    
    for i in range(dims):
        covs.append(np.cov(true_samples[:, :, i]))
    covs = np.array(covs)
    return true_samples, covs
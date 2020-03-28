import numpy as np
from utils import toTuple
import matplotlib.pyplot as plt

def sampleMeasuredSensorFromTrue(dims, num_sensors, target_loc):
    sensor_mus = []
    covs = []
    mu = toTuple(target_loc)
    scalar = 25
    for i in range(num_sensors):
        S = np.tril(np.random.rand(dims, dims))
        cov = np.dot(S, S.T) * scalar
        covs.append(cov)
        mu_i = np.random.multivariate_normal(mu, cov, 1)[0]
        sensor_mus.append(mu_i)

    covs = np.array(covs)
    sensor_mus = np.array(sensor_mus)
    return sensor_mus, covs

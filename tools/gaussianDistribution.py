import numpy as np
from tools.utils import toTuple
import matplotlib.pyplot as plt

def sampleMeasuredSensorFromTrue(dims, num_sensors, target_loc):
    sensor_mus = []
    covs = []
    mu = toTuple(target_loc)
    scalar = 1
    for i in range(num_sensors):
        S = np.tril(np.random.randn(dims, dims))
        cov = np.dot(S, S.T) * scalar
        while(np.linalg.det(cov) < 0.5):
            cov = cov * 2
        covs.append(cov)
        mu_i = np.random.multivariate_normal(mu, cov, 1)[0]
        sensor_mus.append(mu_i)

    covs = np.array(covs)
    for cov in covs:
        verify_pd(cov)
    sensor_mus = np.array(sensor_mus)
    return sensor_mus, covs


def verify_pd(cov):
    eigen_vals = np.linalg.eig(cov)[0]
    for val in eigen_vals:
        if val <= 0:
            print(eigen_vals)
            print(cov)
            raise Exception("Not PD")
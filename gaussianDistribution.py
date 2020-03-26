import numpy as np
from utils import toTuple
import matplotlib.pyplot as plt

def generate_samples(dims, num_samples, num_sensors, target_loc, sens):
    true_samples = np.ndarray(shape=(num_sensors, num_samples, dims))
    covs = []
    fig = plt.figure()
    if(dims >= 3):
        ax= plt.axes(projection='3d')
    else:
        ax = plt.axes()

    for i in range(num_sensors):
        sensor_loc = sens[:, i]
        mu = toTuple(sensor_loc - target_loc)
        S = np.tril(np.random.rand(3, 3))
        cov = S*S.T
        covs.append(cov)
        true_samples[i] = np.random.multivariate_normal(mu, cov, num_samples)
        print("-----------------")
        if(dims >= 3):
            ax.scatter3D(true_samples[i, :, 0], true_samples[i,:,1] , true_samples[i, :, 2])
            ax.scatter3D(mu[0], mu[1], mu[2])
        else:    
            ax.scatter(true_samples[i][0], true_samples[i][1])
            ax.scatter(mu[0], mu[1])

    plt.show()


    covs = np.array(covs)
    return true_samples, covs
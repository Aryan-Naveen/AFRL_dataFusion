from tools.generateGraphHypercube import generateGraphHypercube
from tools.gaussianDistribution import sampleMeasuredSensorFromTrue
from fusionAlgorithms.centralizedFusion import centralizedAlgorithm
from fusionAlgorithms.covarianceIntersection import covarianceIntersection
from fusionAlgorithms.ellipsoidalIntersection import ellipsoidalIntersection
from tools.KL_divergence import compute_KL
from scipy.stats import multivariate_normal
from tools.utils import plot_ellipse, print_all_data

import matplotlib.pyplot as plt
import warnings
import math
import random
import numpy as np


warnings.filterwarnings("ignore")
class Space():
    def __init__(self):
        self.ang_meas_sigma = 0
        self.dim = 0
        self.size_box = 0
        self.border = 0

    def get_figure(self):
        return plt.figure()
    
    def get_axes(self):
        if(self.dim >= 3):
            return plt.axes(projection='3d')
        else:
            return plt.axes()

    def get_3D_axes(self):
        return plt.axes(projection='3d')



def common_rows(matrix, ind):
    occur = []
    for i in range(len(matrix)):
        if(np.all(matrix[i] == matrix[ind]) and not i == ind):
            occur.append(i)
    return occur



my_space = Space()
N_agents = 7
N_samples = 4500
N_max_gmms = 15
P_link = .02
axis_length = 100
KL_inputs = []
N_time_steps = 1000
calculate_KL_guard = False
calculate_covariance_det = True
#Used to plot pdf functions

my_space.ang_meas_sigma = 5 * math.pi/180
my_space.dim = 2

my_space.size_box = axis_length * np.ones((my_space.dim,1))
my_space.border = 10


target_loc = (np.random.rand(my_space.dim, 1) * (my_space.size_box - 2*my_space.border) + my_space.border).reshape((my_space.dim, ))

for val in target_loc:
    #For computation time 
    if(my_space.dim < 3):
        KL_inputs.append(np.linspace(val-5, val+5,num=30))
    else:
        KL_inputs.append(np.linspace(val-2, val+2,num=10))
KL_inputs = np.array(KL_inputs)

x, y = np.mgrid[target_loc[0]-5:target_loc[0]+5:.1, target_loc[1]-5:target_loc[1]+5:.1]
pos = np.empty(x.shape + (2, ))
pos[:, :, 0] = x
pos[:, :, 1] = y


dist = 6
while(dist > 4):
    A, sens = generateGraphHypercube(my_space.dim, N_agents, 0.2)

    sens *= axis_length


    curr_A_pow = np.copy(A)
    sum_A = np.copy(A)
    dist = 1

    while np.count_nonzero(sum_A == 0) > 0:
        curr_A_pow = np.dot(curr_A_pow, A)
        sum_A += curr_A_pow
        dist += 1

print("Diameter of the graph is: " + str(dist))

D = np.sum(A, axis = 0)
max_deg = max(D)
neighbors = {}
for i in range(N_agents):
    neighbors[i] = np.nonzero(A[i])

fig = my_space.get_figure()
ax= my_space.get_axes()

for i in range(1, N_agents):
    for j in range(i):
        if A[i][j] == 1:
            if my_space.dim > 2:
                X = [sens[0][i], sens[0][j]]
                Y = [sens[1][i], sens[1][j]]
                Z = [sens[2][i], sens[2][j]]
                ax.plot(X, Y, Z)
            else:
                X = [sens[0][i], sens[0][j]]
                Y = [sens[1][i], sens[1][j]]
                ax.plot(X, Y)


if my_space.dim >= 3:
    ax.scatter3D(target_loc[0], target_loc[1], target_loc[2])
else:
    ax.scatter(target_loc[0], target_loc[1])

plt.title("Sensor network")
plt.ylabel("Distance (m)")
plt.xlabel("Distance (m)")

plt.savefig("visualizations/SensorNetwork.png")
plt.show()

sensor_mus, sensor_covs = sampleMeasuredSensorFromTrue(my_space.dim, N_agents, target_loc)

# Plot the initial covariances
ax = plt.axes()
for ind, cov in enumerate(sensor_covs):
    plot_ellipse(cov, ax, "Initial " + str(ind + 1))

#Centralized Algorithm
fused_cov, fused_mu = centralizedAlgorithm(np.copy(sensor_covs), np.copy(sensor_mus))
plot_ellipse(fused_cov, ax, "Centralized Fusion", alpha_val=1, color_def='blue')

P_centralized = multivariate_normal(fused_mu, fused_cov)

#CI Decentralized Fusion
sensor_mus_CI, sensor_covs_CI, KL_div_CI, determinants_CI = covarianceIntersection(np.copy(sensor_mus), np.copy(sensor_covs), N_time_steps, neighbors, N_agents, KL_inputs, P_centralized, calculate_KL=calculate_KL_guard, calculate_det=calculate_covariance_det)
plot_ellipse(sensor_covs_CI[0], ax, "Covariance Intersection", alpha_val=1, color_def='purple')

#Ellipsoidal Intersection Fusion
sensor_mus_Ellip, sensor_covs_Ellip, KL_div_Ellip, determinants_Ellip = ellipsoidalIntersection(np.copy(sensor_mus), np.copy(sensor_covs), N_time_steps, neighbors, N_agents, KL_inputs, P_centralized, calculate_KL=calculate_KL_guard, calculate_det=calculate_covariance_det)
print(sensor_covs_Ellip[0])
plot_ellipse(sensor_covs_Ellip[0], ax, "Ellipsoidal Intersection", alpha_val=1, color_def='green')

plt.legend(loc='upper left', borderaxespad=0.)
plt.grid(b = True)
plt.title("Covariance Ellipses")
plt.savefig("visualizations/Covariance Ellipses.png")
plt.show()


output_data = []
output_data.append(["Centralized Fusion", np.linalg.det(fused_cov), "NA", fused_mu])
Q = multivariate_normal(sensor_mus_CI[0], sensor_covs_CI[0])
output_data.append(["Covariance Intersection", np.linalg.det(sensor_covs_CI[0]), compute_KL(P_centralized, Q, KL_inputs), sensor_mus_CI[0]])
Q = multivariate_normal(sensor_mus_Ellip[0], sensor_covs_Ellip[0])
output_data.append(["Ellipsoidal Intersection", np.linalg.det(sensor_covs_Ellip[0]), compute_KL(P_centralized, Q, KL_inputs), sensor_mus_Ellip[0]])
print_all_data(output_data, sensor_covs)




if(calculate_KL_guard):    
    ax = plt.axes()
    for i in range(N_agents):
        kl = KL_div_CI[i]
        ax.plot(X_axis, kl, label="Sensor " + str(i + 1))

    plt.ylabel("KL Divergence")
    plt.xlabel("Time Steps")
    plt.title("KL Divergence CI Progression")
    plt.legend(loc='upper left', borderaxespad=0.)
    plt.grid(b = True)
    plt.savefig("KL_Divergence_CI.png")
    plt.show()
    
    ax = plt.axes()
    for i in range(N_agents):
        kl = KL_div_Ellip[i]
        ax.plot(X_axis, kl, label="Sensor " + str(i + 1))

    plt.ylabel("KL Divergence")
    plt.xlabel("Time Steps")
    plt.title("KL Divergence Ellipsoidal Intersection Progression")
    plt.legend(loc='upper left', borderaxespad=0.)
    plt.grid(b = True)
    plt.savefig("visualizations/KL_Divergence_Ellipse.png")
    plt.show()

if(calculate_covariance_det):
    ax = plt.axes()
    X_axis = [i for i in range(1, min(len(determinants_CI[0]), 50) + 1)]
    for i in range(N_agents):
        deter = determinants_CI[i][:min(len(determinants_CI[i]), 50)]
        ax.plot(X_axis, deter, label = "Sensor " + str(i + 1))

    plt.ylabel("Determinant of Covariance Matrix")
    plt.xlabel("Time Steps")
    plt.title("Covariance Progression CI")
    plt.legend(loc='upper left', borderaxespad=0.)
    plt.grid(b = True)
    plt.savefig("visualizations/Covariance_Progression_CI.png")
    plt.show()




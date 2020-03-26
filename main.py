import numpy as np
from generateGraphHypercube import generateGraphHypercube
from gaussianDistribution import generate_samples
import math

import matplotlib.pyplot as plt

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

my_space.ang_meas_sigma = 5 * math.pi/180
my_space.dim = 3

my_space.size_box = axis_length * np.ones((my_space.dim,1))
my_space.border = 10


target_loc = (np.random.rand(my_space.dim, 1) * (my_space.size_box - 2*my_space.border) + my_space.border).reshape((my_space.dim, ))

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
neighbors = [[] for i in range(N_agents)]
for i in range(len(neighbors)):
    neighbors[i] = np.nonzero(A[i])
neighbors = np.array(neighbors)

fig = my_space.get_figure()
ax= my_space.get_axes()

print(A)
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

plt.show()

true_sample, covariances = generate_samples(my_space.dim, N_samples, N_agents, target_loc, sens)


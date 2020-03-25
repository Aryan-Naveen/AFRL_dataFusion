import numpy as np
import random
import math

def generateGraphHypercube(dims, N_sensors, dist_prob):

    #Constants
    dist_prob_scalar = 1.2

    #Initialize variables
    A = np.zeros((N_sensors, N_sensors))
    locs = np.zeros((dims, N_sensors))
    
    for i in range(N_sensors):
        side = random.randint(0, dims - 1)
        idx = [j for j in range(side)] + [k for k in range(side + 1, dims)]
        locs[side][i] = random.randint(0, 1)
        for val in idx:
            locs[val][i] = random.random()
            
    todo = np.arange(1, N_sensors)

    iter_num = 1
    while (not len(todo) == 0):
        for k in todo:
            if iter_num == 1:
                to_scan = [i for i in range(k)]
            else:
                to_scan = [i for i in range(N_sensors)]
            
            for s in to_scan:
                if not s==k:
                    link_dist = np.linalg.norm(locs[:, k] - locs[:, s])                    
                    prob_link = math.exp(-0.5*(link_dist/dist_prob)**2)
                    if random.random() < prob_link:
                        A[s][k] = 1
                        A[k][s] = 1
        
        todo = np.argwhere(np.sum(A, axis = 0) == 0).reshape(-1)
        iter_num += 1
        dist_prob *= dist_prob_scalar
    

    #Do not completely understand this part
    my_eigs, _ = np.linalg.eig(np.diag(np.sum(A, axis = 0)) - A)
    my_eigs = np.sort(my_eigs)
    while abs(my_eigs[1]) < 1e-1:
        row = random.randint(0, N_sensors - 1)
        col = random.randint(0, N_sensors - 1)
        A[row][col] = 1
        A[col][row] = 1
        my_eigs, _ = np.linalg.eig(np.diag(np.sum(A, axis = 0)) - A)
        my_eigs = np.sort(my_eigs)
        print(my_eigs)
    
    return A, locs


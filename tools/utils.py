import numpy as np
import math
import numpy.linalg as LA
from tabulate import tabulate

def toTuple(arr):
    try:
        return tuple(toTuple(i) for i in arr)
    except TypeError:
        return arr

def plot_ellipse(covariance, ax, label_t):
    if covariance.shape[0] == 2:
        x_el = np.array([np.sin(np.linspace(0, 2*math.pi, num=63)), np.cos(np.linspace(0, 2*math.pi, num=63))])
        C = np.linalg.cholesky(covariance)
        y_el = np.dot(C, x_el)
        ax.plot(y_el[0], y_el[1], label=label_t)

def check_if_singular(M):
    return LA.det(M) == 0

def print_initial_covariances(init_cov):
    print("Initial Covariance Determinant")
    headers = ["Sensor Number", "Intial Covariance"]
    output = [headers]
    for index, cov in enumerate(init_cov):
        output.append([index, LA.det(cov)])
    print(tabulate(output, headers="firstrow"))

def print_output_table(input_data):
    headers = ["Data Fusion Method", "Final_Cov", "KL div", "Final Mu"]
    output = [headers]
    for line in input_data:
        line[-1] = np.matrix.round(line[-1], decimals=1)
        output.append(line)
    print(tabulate(output, headers="firstrow"))

def print_all_data(final_data, intial_covariances):
    print_initial_covariances(intial_covariances)
    print_output_table(final_data)
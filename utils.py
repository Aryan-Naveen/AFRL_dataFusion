import numpy as np
import math

def toTuple(arr):
    try:
        return tuple(toTuple(i) for i in arr)
    except TypeError:
        return arr

def plot_ellipse(covariance, ax, label_t):
    x_el = np.array([np.sin(np.linspace(0, 2*math.pi, num=63)), np.cos(np.linspace(0, 2*math.pi, num=63))])
    C = np.linalg.cholesky(covariance)
    y_el = np.dot(C, x_el)
    ax.plot(y_el[0], y_el[1], label=label_t)

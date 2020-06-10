import numpy as np
import matplotlib.pyplot as plt



def get_bounds_for_nu(e, z, q):
    bounds = [-np.sum(np.power(2*e*z + q, 2)), np.sum(np.power(2*e*z + q, 2))]
    bounds.sort()
    return np.array(bounds)
    

def calculate_value(eigs, q, nu, z, r):
    return np.sum(eigs*np.power(nu*q-2*z, 2)/(4*np.power(1+nu*eigs, 2))) - np.sum((q*(nu*q-2*z))/(2*(1+nu*eigs))) + r


def determine_nu(delta, z, q, r):
    pot_nus = np.linspace(-1000, 1000, 2000)
    print(delta)
    eig = np.diag(delta)
    bounds = get_bounds_for_nu(eig, z, q)
    print(bounds[0])
    print(calculate_value(eig, q, bounds[0], z, r))
    print(bounds[1])
    print(calculate_value(eig, q, bounds[1], z, r))
    print(bounds[0]/2)
    print(calculate_value(eig, q, bounds[0]/2, z, r))

    nu = np.linspace(bounds[0], bounds[1], 2000)
    vals = []
    for n in nu:
        vals.append(calculate_value(eig, q, n, z, r))

    ax = plt.axes()
    ax.plot(nu, vals)
    plt.show()        


def solve_QPQC(z, P, q, r):
    D, Q = np.linalg.eig(P)
    delta_eig = np.diag(D)

    q_hat = Q.T @ q
    z_hat = Q.T @ z

    determine_nu(delta_eig, z_hat, q_hat, r)

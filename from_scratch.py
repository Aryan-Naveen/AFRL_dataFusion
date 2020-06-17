import numpy as np
import matplotlib.pyplot as plt



def get_bounds_for_nu(e):
    bounds = np.zeros((2, 1))
    bounds[1]= -1/np.min(e)
    bounds[0] = -1/np.max(e)
    # print(np.sort(bounds))
    return np.sort(bounds)
    

def calculate_value(eigs, q, nu, z, r):
    A = np.sum(eigs*np.power(nu*q-2*z, 2))
    B = np.sum(4*np.power(1+nu*eigs, 2))

    C = np.sum(2*(1+nu*eigs))
    D = np.sum((q*(nu*q-2*z)))
    return A/B + C/D + r

def calculate_derivative(eig, z, q, nu):
    return -np.sum(np.power(2*eig*z+q, 2)/np.power(2*(1+nu*eig), 3))


def case_1(bounds, eig, q, z, r):
    vfunc = np.vectorize(calculate_value, excluded=['eigs', 'z', 'q'])
    qfunc = np.vectorize(calculate_derivative, excluded=['eig', 'z', 'q'])
    print(bounds)
    nus = np.linspace(-100, 100, 10000)
    vals = np.abs(vfunc(eigs=eig, q=q, nu=nus, z=z, r=r))
    ders = qfunc(eig=eig, z=z, q=q, nu=nus)
    print(ders[1])
    print(ders[2])
    print(ders[3])
    print(ders[4])
    print(ders[5])
    print(ders[6])
    print(ders[7])
    print(ders[8])
    print(ders[9])
    print(ders[10])
    print(ders[11])
    print(ders[12])
    print("...")
    print(ders[-2])
    ax = plt.axes()
    ax.plot(nus, vals)
    plt.show()
    ax = plt.axes()
    ax.plot(nus, ders)
    plt.show()
    a = np.abs(ders)
    # print(ders[np.argwhere(a == np.min(a))][0])
    nu = nus[np.argwhere(a == np.min(a))][0]
    # print(nu)
    # print(-0.5*np.linalg.inv(np.identity(2) + nu*np.diag(eig))@(nu*q-2*z))
    return -0.5*np.linalg.inv(np.identity(2) + nu*np.diag(eig))@(nu*q-2*z)
    

def case_2(bounds, eig, q, z, r):
    for val in bounds:
        A = 2*(np.identity(2) + val*np.diag(eig))
        g = 2*z - val*q
        x_hat = g @ np.linalg.pinv(A)
        # print(x_hat)
        print(x_hat.T @ np.diag(eig) @ x_hat + q.T @ x_hat + r)


def determine_nu(delta, z, q, r, Q):
    pot_nus = np.linspace(-1000, 1000, 2000)
    eig = np.diag(delta)
    bounds = get_bounds_for_nu(eig)
    val1 = calculate_value(eig, q, bounds[0], z, r)
    val2 = calculate_value(eig, q, bounds[1], z, r)
    x_hat = case_1(bounds, eig, q, z, r)
    print(Q @ x_hat)
    return Q @ x_hat
    # case_2(bounds, eig, q, z, r)


    

def solve_QPQC(z, P, q, r):
    D, Q = np.linalg.eig(P)
    delta_eig = np.diag(D)

    q_hat = Q.T @ q
    z_hat = Q.T @ z

    return determine_nu(delta_eig, z_hat, q_hat, r, Q)

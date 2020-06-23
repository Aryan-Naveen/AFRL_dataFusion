import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA

class solve_QPQC_problem():
    def __init__(self, eig, q, z, r):
        self.eig = eig
        self.q = q
        self.z = z
        self.r = r
        self.n = eig.size
    
    def get_bounds_for_nu(self):
        bounds = np.zeros((2, 1))
        eig_min = np.min(self.eig)
        eig_max = np.max(self.eig)
        if eig_min>=0:
            print("WARNING:  There is no upper limit for nu. It can go to +\infty")
        if eig_max <= 0:
            print("WARNING:  nu will have to be negative for I+nu \Lambda to be psd. No limits on it")
        bounds[1]= -1/np.min(self.eig)
        bounds[0] = -1/np.max(self.eig)
        print('bounds before sort',bounds,'and after sort',np.sort(bounds))
        return np.sort(bounds)

    def calculate_value(self, nu):
        A = self.eig*np.power(nu*self.q-2*self.z, 2)
        B = 4*np.power(1+nu*self.eig, 2)

        C = (self.q@(nu*self.q-2*self.z))
        D = 2*(1+nu*self.eig)
        # print(f'eigs is {eigs}, q is {q}, nu is {nu}, z is {z}')
        # print(f'A is {A}, B is {B}, C is {C}, D is {D}, r is {r}')
        return np.sum(A/B - C/D) + self.r

    def calculate_derivative(self, nu):
        return -np.sum(np.power(2*self.eig*self.z+self.q, 2)/np.power(2*(1+nu*self.eig), 3))


    def case_1(self, bounds):
        vfunc = np.vectorize(self.calculate_value)
        qfunc = np.vectorize(self.calculate_derivative)
        print(bounds)
        nus = np.linspace(bounds[0]+1E-3, bounds[1]-1E-3, abs(int((bounds[1]-bounds[0])*1024)))
        vals = vfunc(nus)
        ders = qfunc(nus)
        # print(ders[1])
        # print(ders[2])
        # print(ders[3])
        # print(ders[4])
        # print(ders[5])
        # print(ders[6])
        # print(ders[7])
        # print(ders[8])
        # print(ders[9])
        # print(ders[10])
        # print(ders[11])
        # print(ders[12])
        # print("...")
        # print(ders[-2])
        print('max derivative is',np.max(ders), 'min derivative is',np.min(ders))
        ax = plt.axes()
        ax.plot(nus, vals)
        plt.show()
        ax = plt.axes()
        ax.plot(nus, ders)
        plt.show()
        a = np.abs(vals)
        # print(ders[np.argwhere(a == np.min(a))][0])
        nu = nus[np.argwhere(a == np.min(a))][0][0]
        print("\n\n\n")
        print(nu)
        print("\n\n\n")
        ax = plt.axes()
        ax.plot(nus, vals)
        ax.plot(nu, 0, color='green',  marker='o', linestyle='dashed', linewidth=2, markersize=12)
        plt.show()



        # print(-0.5*np.linalg.inv(np.identity(2) + nu*np.diag(eig))@(nu*q-2*z))
        return -0.5*np.linalg.inv(np.identity(len(self.eig)) + nu*np.diag(self.eig))@(nu*self.q-2*self.z)
    

    def case_2(self):
        e_max = np.max(self.eig)
        e_min = np.min(self.eig)
        pot_nus = []
        if e_max >0:
            pot_nus.append(-1/e_max)
        if e_min < 0:
            pot_nus.append(-1/e_min)
        
        for nu in pot_nus:
            x_hat = np.linalg.pinv(np.identity(2) + nu*np.diag(self.eig))


def determine_nu(delta, z, q, r, Q):

    eig = np.diag(delta)
    # print('eig is \n',eig)
    qcqp = solve_QPQC_problem(eig, q, z, r)

    bounds = qcqp.get_bounds_for_nu()
    x_hat = qcqp.case_1(bounds)
    #These next two will be NAN by definition of the bounds...

    # print('val1 is ',val1,'and val2 is',val2)

    print("XHAT")
    print(x_hat)
    print("\n\n\n")
    # print(Q @ x_hat)
    return Q @ x_hat
    # case_2(bounds, eig, q, z, r)


    

def solve_QPQC(z, P, q, r):
    D, Q = np.linalg.eigh(P)
    delta_eig = np.diag(D)

    print("\n\n\n\n")
    print(Q.T @ Q)
    print("\n\n\n\n")

    q_hat = Q.T @ q
    z_hat = Q.T @ z

    return determine_nu(delta_eig, z_hat, q_hat, r, Q)


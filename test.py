from QCQP_opt.qcqp_solver import QCQP_solver
import numpy as np
from from_scratch import solve_QPQC

def test_quadratic_programming():
    print("A simple test direct qcqp...")
    z = np.array([1.,0.])
    P = np.array([[2,0.],[0.,-1]])
    q = np.array([1.,0])
    r = 0
    solver = QCQP_solver(P, q, r, z)
    solver.P = P
    solver.q = q
    solver.z = z
    solver.r = r
    solver.eig = np.diag(P)
    solver.generate_bounds_for_nu()
    nu = solver.find_optimal_nu()
    print(solver.calculate_x_c_val(nu))


def test_eigen_val_transform():
    print("A simple test direct qcqp...")
    z = np.random.randn(1, 2)[0]
    P = np.diag(np.random.randn(1, 2)[0])
    print(P)
    q = np.random.randn(1, 2)[0]
    r = 0
    solver = QCQP_solver(P, q, r, z)
    solver.P = P
    solver.q = q
    solver.z = z
    solver.r = r

    solver.perform_eigen_transform()

    print(P == solver.P)
    print(z == solver.z)
    print(q == solver.q)

def test_cholseky_transform():
    print("A simple test direct qcqp...")
    z = np.random.randn(1, 2)[0]
    P = np.diag([5, 5])
    print(P)
    q = np.random.randn(1, 2)[0]
    r = 0
    solver = QCQP_solver(P, q, r, z)
    solver.perform_cholesky_transform(P)
    print(solver.P == np.identity(2))
    print(solver.P)



    solver.perform_eigen_transform()

    print(np.identity(2) == solver.P)
    print(solver.P)
    S = np.linalg.cholesky(P).T
    print(S @ z == solver.z)


test_quadratic_programming()
test_eigen_val_transform()
test_cholseky_transform()
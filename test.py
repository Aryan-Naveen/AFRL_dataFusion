from QCQP_opt.qcqp_solver import QCQP_solver
from QCQP_opt.qcqp_solver import calculate_mahalonobis_difference
import numpy as np
from from_scratch import solve_QPQC

def get_random_psd():
    A = np.random.randn(2, 2)
    return A.T @ A

def get_random_vectors():
    return np.random.randn(1, 2)[0]

def get_random_matrix():
    return np.random.randn(2, 2)

def test_quadratic_programming():
    print("A simple test direct qcqp...")
    z = get_random_vectors()
    P = get_random_psd() - get_random_psd()
    q = get_random_vectors()
    r = 0
    solver = QCQP_solver(P, q, r, z)
    solver.perform_eigen_transform()
    solver.generate_bounds_for_nu()
    nu = solver.find_optimal_nu()
    x_c = solver.inverse_eigen_transform(solver.calculate_x_c_val(nu))

def test_eigen_val_transform():
    #~~~~~~~~~~~~~~~~~~~~~~~~~INITIALIZATION~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    z = get_random_vectors()
    x = get_random_vectors()
    P = get_random_psd() - get_random_psd()
    q = get_random_vectors()
    r = 0
    solver = QCQP_solver(P, q, r, z)
    #~~~~~~~~~~~~~~~~~~~~~~~~~INITIAL CALCULATIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print("~~~~~~~~~~~~~~~~~~~~~~~~~INITIAL CALCULATIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("MAHALANOBIS DIFFERENCE: " + str(calculate_mahalonobis_difference(x, z, solver.P)))
    print("INITIAL CONSTRAINT: " + str(solver.calculate_constraint(x)))
    #~~~~~~~~~~~~~~~~~~~~~~~~~TRANSFORM~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    solver.perform_eigen_transform()
    x = solver.Q.T @ x
    #~~~~~~~~~~~~~~~~~~~~~~~~~FINAL CALCULATIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print("~~~~~~~~~~~~~~~~~~~~~~~~~FINAL CALCULATIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("MAHALANOBIS DIFFERENCE: " + str(calculate_mahalonobis_difference(x, solver.z, solver.P)))
    print("INITIAL CONSTRAINT: " + str(solver.calculate_constraint(x)))
    #~~~~~~~~~~~~~~~~~~~~~~~~~INVERSE TRANSFORM~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    x = solver.inverse_eigen_transform(x)
    #~~~~~~~~~~~~~~~~~~~~~~~~~ORIGINAL CALCULATIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print("~~~~~~~~~~~~~~~~~~~~~~~~~INVERSE CALCULATIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("MAHALANOBIS DIFFERENCE: " + str(calculate_mahalonobis_difference(x, solver.z, solver.P)))
    print("INITIAL CONSTRAINT: " + str(solver.calculate_constraint(x)))



def test_cholseky_transform():
    #~~~~~~~~~~~~~~~~~~~~~~~~~INITIALIZATION~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    z = get_random_vectors()
    x = get_random_vectors()
    P = get_random_psd()
    q = get_random_vectors()
    r = 0
    solver = QCQP_solver(P, q, r, z)
    #~~~~~~~~~~~~~~~~~~~~~~~~~INITIAL CALCULATIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print("~~~~~~~~~~~~~~~~~~~~~~~~~INITIAL CALCULATIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("MAHALANOBIS DIFFERENCE: " + str(calculate_mahalonobis_difference(x, z, solver.P)))
    print("INITIAL CONSTRAINT: " + str(solver.calculate_constraint(x)))
    #~~~~~~~~~~~~~~~~~~~~~~~~~TRANSFORM~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    solver.perform_cholesky_transform(P)
    x = solver.S @ x
    #~~~~~~~~~~~~~~~~~~~~~~~~~FINAL CALCULATIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print("~~~~~~~~~~~~~~~~~~~~~~~~~FINAL CALCULATIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("MAHALANOBIS DIFFERENCE: " + str(calculate_mahalonobis_difference(x, solver.z, solver.P)))
    print("INITIAL CONSTRAINT: " + str(solver.calculate_constraint(x)))
    #~~~~~~~~~~~~~~~~~~~~~~~~~INVERSE TRANSFORM~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    x = solver.inverse_cholseky(x)
    #~~~~~~~~~~~~~~~~~~~~~~~~~ORIGINAL CALCULATIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print("~~~~~~~~~~~~~~~~~~~~~~~~~INVERSE CALCULATIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("MAHALANOBIS DIFFERENCE: " + str(calculate_mahalonobis_difference(x, solver.z, solver.P)))
    print("INITIAL CONSTRAINT: " + str(solver.calculate_constraint(x)))



print("############### SiMPLE TEST CASE ####################")
test_quadratic_programming()
print("\n")
print("############### EIGEN VALUE TEST CASE ####################")
test_eigen_val_transform()
print("\n")
print("############### CHOLESKY VALUE TEST CASE ####################")
test_cholseky_transform()
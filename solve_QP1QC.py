import numpy as np
import math
import numpy.linalg as LA
from tools.gaussianDistribution import verify_pd
from fusionAlgorithms.inverseCovarianceIntersection import ICI
from tools.utils import plot_ellipse
import matplotlib.pyplot as plt
from fusionAlgorithms.EllipsoidalKT import EllipsoidalIntersection
from numpy.random import randn
from from_scratch import solve_QPQC


def psuedo_invert_diagonal_matrix(M):
    diag_M = np.diag(M)
    M_psuedo_inv_diagonal = np.where(diag_M != 0, 1/diag_M, 0)
    M_psuedo_inv = np.diag(M_psuedo_inv_diagonal)
    return M_psuedo_inv


def to_diag_mat(vec):
    return np.diag(vec)

def drop_off_diagonals(mat):
    return to_diag_mat(np.diag(mat))

def isDiag(M):
    return np.allclose(np.diag(np.diag(M)), M)

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def sim_diag(M1,M2, eigenM1=None, eigenM2=None, tol = 10^-4, return_diags=False):
    if(not check_symmetric(M1)):
        raise Exception("M1 must be symmetric")
    if(not check_symmetric(M2)):
        raise Exception("M2 must be symmetric")
    
    if np.any(eigenM1[0] <= 0):
        if np.any(eigenM2[0] <= 0):
            raise Exception("Neither B nor A are pd")
        else:
            print("M1 and M2 are swapped")
            out_pre = sim_diag(M2, M1, eigenM2, eigenM1, tol=tol, return_diags=True)
            if(not return_diags):
                return out_pre
            out = out_pre
            out["M1diag"] = out_pre["M2diag"]
            out["M2diag"] = out_pre["M1diag"]
            return out
    
    a = np.diag([1.0/math.sqrt(val) for val in eigenM1[0]])
    sqrtInvM1 = np.dot(eigenM1[1], a)
    Z = sqrtInvM1.T @ M2 @ sqrtInvM1
    Z = (Z + Z.T)/2.0
    eigZvals, eigZvecs = np.linalg.eig(Z)
    Q = sqrtInvM1 @eigZvecs
    Q_inv = eigZvecs.T @ np.diag([math.sqrt(val) for val in eigenM1[0]]) @ eigenM1[1]
    inv_error = np.max(np.abs(Q_inv @ Q - np.identity(M1.shape[1])))

    output = {}
    output["diagonalizer"] = Q
    output["inverse_diagonalizer"] = Q_inv
    output["M1diag"] = np.array([1]*M1.shape[1])
    output["M2diag"] = np.diag(Q.T @ M2 @ Q)
    if (not return_diags):
        return Q
    return output

def set_QP_unconstrained_eq_0(M, v, k, tol):
    feasible = False
    soln = None
    value = None

    diag_M = np.diag(M)
    M_psuedo_inv = psuedo_invert_diagonal_matrix(M)
    x0 = -0.5*M_psuedo_inv @ v
    def eval_x(x):
        return np.sum(x**2*diag_M) + sum(x*v) + k
    
    eval_x0 = eval_x(x0)
    feasible = eval_x0==0
    if(feasible):
        output = {}
        output["feasible"] = True
        output["soln"] = x0
        output["value"] = eval_x0
        return output
    
    move_ind = None

    v_candidates = np.logical_and(diag_M==0, v!=0)
    v_cand_ind = np.where(v_candidates)[0]

    if(np.sum(v_candidates) > 0):
        move_ind = np.where(v==max([abs(v[i]) for i in v_cand_ind]))[0]        
    elif(eval_x0 < 0 and max(diag_M) > 0):
        move_ind = np.where(diag_M == max(diag_M))[0]
    elif(eval_x0 > 0 and min(diag_M) < 0):
        move_ind = np.where(diag_M == min(diag_M))[0]

    if(move_ind != None):
        soln = x0
        soln_base = x0
        soln_base[move_ind] = 0
        coeffs = [eval_x(soln_base), v[move_ind], diag_M[move_ind]]
        move_complex = np.roots(coeffs)[0]
        if(move_complex.imag > tol):
            raise Exception("error in root find")
        if(abs(eval_x(soln)) > tol):
            raise Exception('calculation error')
        feasible = True
    
    output = {}
    output["feasible"] = feasible
    output["soln"] = soln
    output["value"] = eval_x(soln)
    return output

def min_QP_unconstrained(M, v, tol):
    print(tol)
    BIG: float = (1.0/tol)**4
    print(BIG)
    if(not (np.all(np.isfinite(M)) or np.isfinite(v))):
        raise Exception("M and v must be finite")
    if(not isDiag(M)):
        raise Exception("M should be diagonal")
    if(len(v) != M.shape[0]):
        raise Exception("dimension of M and v must match")
    
    diag_M = np.diag(M)
    zero_directions = np.logical_and(diag_M == 0, np.array(v==0))
    moves_up_to_pos_Inf = np.logical_or(np.logical_and(diag_M==0, np.array(v > 0)), diag_M > 0)
    moves_up_to_neg_Inf = np.logical_or(np.logical_and(diag_M==0, v < 0), diag_M < 0)
    moves_down_to_pos_Inf = np.logical_or(np.logical_and(diag_M==0, np.array(v < 0)), diag_M > 0)
    moves_down_to_neg_Inf = np.logical_or(np.logical_and(diag_M==0, np.array(v > 0)), diag_M < 0)
    if(np.any(np.logical_or((moves_down_to_pos_Inf + moves_down_to_neg_Inf + zero_directions).astype(int) != 1, (moves_up_to_pos_Inf + moves_up_to_neg_Inf + zero_directions).astype(int) !=1))):
        raise Exception("direction error")

    finite_soln = not np.any(np.logical_or(moves_down_to_neg_Inf, moves_up_to_neg_Inf))

    if(finite_soln):
        M_psuedo_inv = psuedo_invert_diagonal_matrix(M)
        soln = -0.5*M_psuedo_inv @ v
    else:
        soln = np.array([0.0]*len(v))
        soln.astype("float64")
        soln[moves_up_to_neg_Inf] = BIG
        soln[moves_down_to_neg_Inf] = BIG
    
    soln.reshape(len(v), 1)
    print(soln)
    print(v)
    value = soln.T @ M @ soln + np.cross(v.T, soln.T)

    output = {}
    output["zero_directions"] = zero_directions
    output["moves_up_to_pos_Inf"] = moves_up_to_pos_Inf
    output["moves_up_to_neg_Inf"] = moves_up_to_neg_Inf
    output["moves_down_to_pos_Inf"] = moves_down_to_pos_Inf
    output["moves_down_to_neg_Inf"] = moves_down_to_neg_Inf
    output["finite_soln"] = finite_soln
    output["value"] = value
    output["soln"] = soln

    return output


#' Solve (non-convex) quadratic program with 1 quadratic constraint
#' 
#' Solves a possibly non-convex quadratic program with 1 quadratic constraint. Either \code{A_mat} or \code{B_mat} must be positive definite, but not necessarily both (see Details, below).
#' 
#' Solves a minimization problem of the form:
#' 
#' \deqn{ 	min_{x} x^T A_mat x + a_vec^T x }
#' \deqn{ such that x^T B_mat x + b_vec^T x + k \leq 0,}
#' 
#' where either \code{A_mat} or \code{B_mat} must be positive definite, but not necessarily both.
#' 
#' @param A_mat see details below
#' @param a_vec see details below
#' @param B_mat see details below
#' @param b_vec see details below
#' @param k see details below
#' @param tol a calculation tolerance variable used at several points in the algorithm.
#' @param eigen_A_mat (optional) the precalculated result \code{eigen(A_mat)}.
#' @param eigen_B_mat (optional) the precalculated result \code{eigen(B_mat)}.
#' @param verbose show progress from calculation
#' @import quadprog
#' @return a list with elements
#' \itemize{
#'	\item{soln}{ - the solution for x}
#'	\item{constraint}{ - the value of the constraint function at the solution}
#'	\item{objective}{ - the value of the objective function at the solution}
#' }
def solve_QP1QC(A_mat, a_vec, B_mat, b_vec, k, tol=10^-7, verbose= True):
    print(tol)
    eigen_B_mat = np.linalg.eig(B_mat)
    eigen_A_mat = np.linalg.eig(A_mat)
    print(eigen_B_mat)
    print(eigen_A_mat)
    if np.any(eigen_B_mat[0] <= 0):
        if np.any(eigen_A_mat[0] <= 0):
            raise Exception("Neither B nor A are pd")
    sdb = sim_diag(B_mat, A_mat, eigenM1=eigen_B_mat, eigenM2=eigen_A_mat, tol=tol, return_diags=True)
    Q = sdb["diagonalizer"]
    B = Q.T @ B_mat @ Q
    b = Q.T @ b_vec
    A = Q.T @ A_mat @ Q
    a = Q.T @ a_vec

    A = drop_off_diagonals(A)
    B = drop_off_diagonals(B)
    Bd = np.diag(B)
    Ad = np.diag(A)

    def calc_diag_Lagr(x, nu):
        return np.sum(x**2*np.diag(A)) + np.cross(x.T, (a + nu*b).T)
    
    def calc_diag_obj(x):
        return np.sum(x**2 * np.diag(A)) + np.cross(x.T, a.T)
    
    def calc_diag_constraint(x):
        return np.sum(x**2*np.diag(B)) + np.cross(x.T, b.T) + k
    
    def return_on_original_space(x):
        soln = {}
        soln["soln"] = Q @ x
        soln["constraint"] = calc_diag_constraint(x)
        soln["objective"] = calc_diag_obj(x)
        return soln
    

    def test_nu_pd(nu):
        soln = -(0.5) * ((Ad+nu*Bd)**-1)@(a + nu*b)
        constraint_val = calc_diag_constraint(soln)
        out = ""
        if(constraint_val > 0):
            out = "low"
        elif(constraint_val < 0):
            out = "high"
        else:
            out = "optimal"
        
        output = {}
        output["soln"] = soln
        output["type"] = out
        return output
    

    def test_nu_psd(nu, tol):
        diag_A_nu_B = Ad + nu*Bd
        if(np.any(diag_A_nu_B < -tol)):
            raise Warning("Possible error in pd")
        
        diag_A_nu_B = np.where(diag_A_nu_B<0, 0, diag_A_nu_B)
        mat_A_nu_B = to_diag_mat(diag_A_nu_B)
        if(np.all(diag_A_nu_B > 0)):
            raise Exception("error in psd")
        
        I_nu = np.where(diag_A_nu_B>0)
        N_nu = np.where(diag_A_nu_B==0)

        B_d_val = np.take(B, N_nu)
        non_opt_value = None
        
        if(np.any(B_d_val > 0) and np.any(B_d_val < 0)):
            return "optimal"
        elif(np.any(B_d_val < 0)):
            non_opt_value = "high"
        elif(np.any(B_d_val) > 0):
            non_opt_value = "low"
        else:
            raise Exception("PSD TEST SHOULD NOT HAVE BEEN CALLED")
        

        if(np.max(np.abs(np.take(a + nu*b, N_nu))) > 0):
            out = {}
            out["type"] = non_opt_value
            out["soln"] = None
            return out
        
        A_nu_B_psuedo_inv = psuedo_invert_diagonal_matrix(mat_A_nu_B)
        x_I = (-0.5)*A_nu_B_psuedo_inv @ (a + nu*b)
        if(np.any(np.array([x_I[i] for i in N_nu]) != 0)):
            raise Exception("indexing error")

        free_constr_opt = set_QP_unconstrained_eq_0(B[N_nu][:, N_nu], b[N_nu], calc_diag_constraint(x_I), tol)
        if(free_constr_opt["feasible"]):
            soln = x_I
            soln[N_nu] = free_constr_opt
            output = {}
            output["type"] = "optimal"
            output["soln"] = soln
            return output
        else:
            output = {}
            output["type"] = non_opt_value
            output["soln"] = None
            return output
        
        #Optimality check #2

    def test_nu(nu, tol):
        if(np.any((Ad + nu*Bd) < -tol)):
            raise Exception("invalid nu")
        if(np.all(Ad + nu * Bd > 0)):
            return test_nu_pd(nu)
        else:
            return test_nu_psd(nu, tol)

    constr_prob = min_QP_unconstrained(B, b, tol)
    if(constr_prob["finite_soln"]):
        if(constr_prob["value"] > -k):
            raise Exception("Constraint is not feasible")
    if(constr_prob["value"] == -k):
        raise Warning("Constraint may be too strong")
    
    u_prob = min_QP_unconstrained(A, a, tol)

    def min_constr_over_restricted_directions(directions):
        # if(len(directions) == 0):
            #return u_soln
        print(tol)
        search_over_free_elements = min_QP_unconstrained(to_diag_mat(Bd[directions]), b[directions], tol)

        u_soln = u_prob["soln"]
        u_soln[directions] = search_over_free_elements["soln"]
        return u_soln

    u_soln = u_prob["soln"]
    if(u_prob["finite_soln"]):
        if(np.any(u_prob["zero_directions"])):
            u_soln = min_constr_over_restricted_directions(u_prob["zero_directions"])
    
    if(calc_diag_constraint(u_soln) <= 0):
        if(verbose):
            print("Unconstrained solution also solves problem")
            return(return_on_original_space(u_soln))

    nu_opt = None
    x_opt = None
    nu_to_check = []
    nu_max = np.inf
    nu_min = -np.inf
    if(np.any(Bd > 0)):
        nu_min = np.max((-Ad/Bd)[Bd > 0])
        nu_to_check.append(nu_min)
    if(np.any(Bd < 0)):
        nu_max = np.min((-Ad/Bd)[Bd < 0])
        nu_to_check.append(nu_max)
    
    if(len(nu_to_check) == 0):
        if(np.any(Bd != 0)):
            raise Exception("Error in Bd check")
        if(np.any(B_mat != 0)):
            raise Exception("Error in Bd check")
        if(np.any(Ad <= 0)):
            raise Exception("Error in Pd check")

        raise Exception("Quadratic constraint not active")

    for i in range(len(nu_to_check)):
        print(nu_to_check[i])
        test_nu_check = test_nu(nu_to_check[i], tol)
        if(test_nu_check["type"] == 'optimal'):
            nu_opt = nu_to_check[i]
    
    if(np.isinf(nu_max)):
        nu_max = abs(nu_min)
        test_nu_max = test_nu(nu_max, tol)
        counter = 0
        while(abs(nu_max) < 1/tol and test_nu_max["type"]== 'low'):
            nu_max = abs(nu_max) * 10
            test_nu_max = test_nu(nu_max, tol)
            counter += 1
            if(counter > 1000):
                raise Exception('infinite loop')
        if(test_nu_max["type"] == 'low'):
            nu_opt = nu_max
            raise Warning("outer limit reached")
    
    if(np.isinf(-nu_min)):
        nu_min = -abs(nu_max)
        test_nu_min = test_nu(nu_min, tol)
        counter = 0
        while(abs(nu_min) < 1/tol and test_nu_max["type"]== 'high'):
            nu_min = abs(nu_min) * 10
            test_nu_min = test_nu(nu_min, tol)
            counter += 1
            if(counter > 1000):
                raise Exception('infinite loop')
        if(test_nu_min["type"] == 'high'):
            nu_opt = nu_min
            raise Warning("outer limit reached")
    
    if(nu_opt == None):
        def bin_search(nu):
            tested_type = test_nu(nu, tol)["type"]
            if(tested_type == 'high'):
                return 1
            elif(tested_type=='low'):
                return -1
            else:
                return 0
        def bin_search_tol(tol, range):
            tested_type = None
            count = 0
            while(not tested_type == 0 or count < 50):
                nu = 0.5*(range[0] + range[1])
                if(tested_type == -1):
                    range[0] = nu
                elif(tested_type == 1):
                    range[1] = nu
                count += 1
            
            return nu
        nu_opt=bin_search_tol(tol, [nu_min, nu_max])
    
    x_opt = test_nu(nu_opt, tol)["soln"]
    return return_on_original_space(x_opt)


def performFusionProbablistic(mu_a, C_a, mu_b, C_b):
    dims = 2


    ei = EllipsoidalIntersection()

    plt.cla()
    plt.clf()
    C_c = ei.mutual_covariance(C_a, C_b) + 1e-1*np.identity(2)
    ax = plt.axes()
    plot_ellipse(C_c, ax, "Common", color_def="orange")
    plot_ellipse(C_a, ax, "A")
    plot_ellipse(C_b, ax, "B")
    plt.show()
    C_ac_inv = LA.inv(C_a) - LA.inv(C_c)
    C_bc_inv = LA.inv(C_b) - LA.inv(C_c)

    ax = plt.axes()
    plot_ellipse(C_c, ax, "Common", color_def="orange")
    plot_ellipse(C_a, ax, "A")
    plot_ellipse(C_b, ax, "B")
    plot_ellipse(LA.inv(C_a), ax, "A inv", color_def="green")
    plot_ellipse(LA.inv(C_b), ax, "B inv", color_def="green")
    plot_ellipse(LA.inv(C_c), ax, "Common inv", color_def="blue")
    # plot_ellipse(LA.inv(C_ac_inv), ax, "CACINV", color_def="blue")
    plt.show()

    # print(LA.eig(C_ac_inv))
    # verify_pd(LA.inv(C_ac_inv))
    # verify_pd(LA.inv(C_bc_inv))

    K_a = LA.inv(C_a) @ (LA.inv(C_ac_inv)@LA.inv(C_a) - np.identity(C_a.shape[0]))
    K_b = LA.inv(C_b) @ (LA.inv(C_bc_inv)@LA.inv(C_b) - np.identity(C_b.shape[0]))

    S = np.linalg.cholesky(K_a)

    B_mat = K_a - K_b
    b_vec = -2*(mu_a.T @ K_a - mu_b.T @ K_b)
    q = mu_a.T @ K_a @ mu_a - mu_b.T @ K_b @ mu_b

    print(B_mat)
    print(b_vec)
    print(q)

    x_c = solve_QPQC(S @ mu_a, B_mat, b_vec, q)
    x_c_a = np.linalg.inv(S) @ x_c
    print("===================")
    print("RESULTS:")
    print("X_c " + str(np.linalg.inv(S) @ x_c))
    print("MAHALONOBIS DIFFERENCE TO A " + str((mu_a-x_c_a).T @ K_a @ (mu_a - x_c_a)))
    print("MAHALONOBIS DIFFERENCE TO B " + str((mu_b - x_c_a).T @ K_b @ (mu_b - x_c_a))) 

    # print("------------------------")
    # B_mat = K_a - K_b
    # b_vec = -2*(mu_a.T @ K_a - mu_b.T @ K_b)
    # q = mu_a.T @ K_a @ mu_a - mu_b.T @ K_b @ mu_b
    # solve_QPQC(mu_a, B_mat, b_vec, q)

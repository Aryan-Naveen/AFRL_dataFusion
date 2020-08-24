from tqdm import tqdm
import numpy as np
from scipy.stats import invwishart as iw        
import matplotlib.pyplot as plt

def inv(A):
    return np.linalg.inv(A)

def calculate_MSE(true_x_fus, C_a, C_b, C_c, x_a, x_b, C_fus):
    C_ac_inv = inv(C_a) - inv(C_c)
    C_ac = inv(C_ac_inv)
    C_bc_inv = inv(C_b) - inv(C_c)
    C_bc = inv(C_bc_inv)
    C_c_inv = inv(C_c)
    
    
    C_abc_inv_inv = inv(C_ac_inv + C_bc_inv)
    
    
    x_c = (C_abc_inv_inv @ (C_ac_inv @ x_a.T + C_bc_inv @ x_b.T)).T
    x_ac = (C_ac @ (inv(C_a) @ x_a.T - C_c_inv @ x_c.T)).T
    x_bc =(C_bc @ (inv(C_b) @ x_b.T - C_c_inv @ x_c.T)).T
    
    x_fus = C_fus @ (C_ac_inv @ x_ac.T + C_bc_inv @ x_bc.T + C_c_inv @ x_c.T)
    
    mse = (np.square(true_x_fus - x_fus)).mean()
    return mse
    
    

def run_sim(trials, df):
    EI = []
    ei_mse = []
    PC_10 = []
    PC_10_mse = []
    PC_05 = []
    PC_05_mse = []
    for i in tqdm(range(trials)):
        import warnings
        warnings.filterwarnings("ignore")
        import numpy as np
        from scipy.optimize import minimize
        from scipy.stats import chi2
        from scipy.linalg import sqrtm
        from numpy.linalg import det
        import numpy.linalg as LA
        import matplotlib.pyplot as plt
        import math

        debug= True

        def generate_covariance(true_mu, dims, df):
            S = (np.tril(iw.rvs(df, 1, size=dims**2).reshape(dims, dims)))*df
            cov = np.dot(S, S.T)
            while(abs(np.linalg.det(cov)) < 1.5):
                cov = cov + 0.5*np.diag(np.diag(cov))
            mu = np.random.multivariate_normal(true_mu, cov, 1)[0]

            return mu, cov

        def get(dims, df):
            true_mu = np.zeros((dims, ))

            x_ac, C_ac = generate_covariance(true_mu, dims, df)
            x_c, C_c = generate_covariance(true_mu, dims, df)
            x_bc, C_bc = generate_covariance(true_mu, dims, df)

            C_a = LA.inv(LA.inv(C_ac) + LA.inv(C_c))
            C_b = LA.inv(LA.inv(C_bc) + LA.inv(C_c))

            x_a = C_a @ (LA.inv(C_ac) @ x_ac + LA.inv(C_c) @ x_c)
            x_b = C_b @ (LA.inv(C_bc) @ x_bc + LA.inv(C_c) @ x_c)

            C_fus = LA.inv(LA.inv(C_a) + LA.inv(C_b) - LA.inv(C_c))
            
            x_fus = C_fus @ (LA.inv(C_ac) @ x_ac + LA.inv(C_bc) @ x_bc + LA.inv(C_c) @ x_c)

            return x_a.reshape(1, dims), x_b.reshape(1, dims), C_a, C_b, C_fus, x_fus

        def plot_ellipse(covariance, ax, label_t="", linestyle='', alpha_val=0.25, color_def='red', center = [0, 0]):
            if covariance.shape[0] == 2:
                x_el = np.array([np.sin(np.linspace(0, 2*math.pi, num=63)), np.cos(np.linspace(0, 2*math.pi, num=63))])
                C = np.linalg.cholesky(covariance)
                y_el = np.dot(C, x_el)
                if len(linestyle) > 0:
                    if len(label_t) > 0:
                        ax.plot(y_el[0] + center[0], y_el[1] + center[1], label=label_t, alpha=alpha_val, color=color_def, linestyle=linestyle)
                    else:
                        ax.plot(y_el[0] + center[0], y_el[1] + center[1], alpha=alpha_val, color=color_def, linestyle=linestyle)            
                else:
                    if len(label_t) > 0:
                        ax.plot(y_el[0] + center[0], y_el[1] + center[1], label=label_t, alpha=alpha_val, color=color_def)
                    else:
                        ax.plot(y_el[0] + center[0], y_el[1] + center[1], alpha=alpha_val, color=color_def)            

        def mutual_covariance(cov_a, cov_b):
            D_a, S_a = np.linalg.eigh(cov_a)
            D_a_sqrt = sqrtm(np.diag(D_a))
            D_a_sqrt_inv = inv(D_a_sqrt)
            M = np.dot(np.dot(np.dot(np.dot(D_a_sqrt_inv, inv(S_a)), cov_b), S_a), D_a_sqrt_inv)    # eqn. 10 in Sijs et al.
            D_b, S_b = np.linalg.eigh(M)
            D_gamma = np.diag(np.clip(D_b, a_min=1.0, a_max=None))   # eqn. 11b in Sijs et al.
            return np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(S_a, D_a_sqrt), S_b), D_gamma), inv(S_b)), D_a_sqrt), inv(S_a))  # eqn. 11a in Sijs et al

        def get_critical_value(dimensions, alpha):
            return chi2.ppf((1 - alpha), df=dimensions)

        def inv(mat):
            return np.linalg.inv(mat)

        def objective(S):
            return -(S[0]*S[3])
            # S = S.reshape(2, 2).T
            # C_c = inv(S @ S.T)
            # return np.trace(C_c)

        def constraint1(S):
            S = S.reshape(2, 2).T
            A = inv(C_a) - S@S.T
            return np.linalg.eig(A)[0][0]
        def constraint2(S):
            S = S.reshape(2, 2).T
            A = inv(C_a) - S@S.T
            return np.linalg.eig(A)[0][1]
        def constraint3(S):
            S = S.reshape(2, 2).T
            A = inv(C_b) - S@S.T
            return np.linalg.eig(A)[0][0]
        def constraint4(S):
            S = S.reshape(2, 2).T
            A = inv(C_b) - S@S.T
            return np.linalg.eig(A)[0][1]

        def psuedoinv(A):
            A[np.where(A<=1e-5)] = 1e-5

        def relu(v):
            threshold = 1E-5
            if v < 100 and v > threshold:
                return np.log1p(1 + np.exp(v))* threshold /np.log1p(1+np.exp(threshold))
            else:
                return v



        def pinv(A):
            RELU = np.vectorize(relu)
            tmp_eig, tmp_egv = LA.eig(A)
            M_inv = tmp_egv @ np.diag(1/RELU(tmp_eig)) @ tmp_egv.T
            M = tmp_egv @ np.diag(RELU(tmp_eig)) @ tmp_egv.T
            return M, M_inv

        def prob_constraint(S):
            S = S.reshape(2, 2).T
            C_c_inv = S@S.T

            C_ac_inv, C_ac = pinv(inv(C_a) - C_c_inv)
            C_bc_inv, C_bc = pinv(inv(C_b) - C_c_inv)

            _, C_abc_inv_inv = pinv(C_ac_inv + C_bc_inv)
            _, C_abc_inv = pinv(C_ac + C_bc)

            x_c = (C_abc_inv_inv @ (C_ac_inv @ x_a.T + C_bc_inv @ x_b.T)).T
            x_ac = (C_ac @ (inv(C_a) @ x_a.T - C_c_inv @ x_c.T)).T
            x_bc =(C_bc @ (inv(C_b) @ x_b.T - C_c_inv @ x_c.T)).T
            f = ((x_ac - x_bc) @ C_abc_inv @ (x_ac - x_bc).T)[0][0]
            return eta - f

        def constraint5(S):
            return round(S[2], 10)

        #inside each loop, first create the input distributions

        x_a, x_b, C_a, C_b, C_fus, t_x_fus = get(2, df)
        # ax = plt.axes()
        # plot_ellipse(C_a, ax, alpha_val=1, color_def="orange", linestyle="dashed")
        # plot_ellipse(C_b, ax, alpha_val=1, color_def="blue", linestyle="dashed")
        # plt.show()
        

        x_a = x_a.reshape(1, 2)
        x_b = x_b.reshape(1, 2)

        ## Now setup the constraints

        con1 = {'type': 'ineq', 'fun': constraint1}
        con2 = {'type': 'ineq', 'fun': constraint2}
        con3 = {'type': 'ineq', 'fun': constraint3}
        con4 = {'type': 'ineq', 'fun': constraint4}
        con5 = {'type': 'ineq', 'fun': prob_constraint}
        con6 = {'type': 'eq', 'fun': constraint5}
        cons = [con1, con2, con3, con4, con5, con6]

        eta = get_critical_value(2, 0.05)
        print('eta1 is ',eta)

        #Find an initialization point
        S_0 = 0.4*(np.linalg.cholesky(inv(mutual_covariance(C_a, C_b))).T).reshape(4, )
        prob_constraint(S_0)

        if(debug):
            print('Before optimization, we have...')
            print('C_a\n',C_a)
            print('C_b\n',C_b)
            print ('objective is',objective(S_0))
            print ('constraint1 is ',constraint1(S_0))
            print ('constraint2 is ',constraint2(S_0))
            print ('constraint3 is ',constraint3(S_0))
            print ('constraint4 is ',constraint4(S_0))
            print ('constraint5 is ',constraint5(S_0))
            print ('prob_constraint is ',prob_constraint(S_0))

        # Run the optimization
        sol = minimize(objective, S_0, method='trust-constr', constraints=cons)
        # print(sol)
        S = sol.x
        S = S.reshape(2, 2).T

        C_c_05 = inv(S.T) @ inv(S)
        if(debug):
            print ('for .05, after optimization we have...')
            print('C_c_05',C_c_05)
            print ('objective is',objective(sol.x))
            print ('constraint1 is ',constraint1(sol.x))
            print ('constraint2 is ',constraint2(sol.x))
            print ('constraint3 is ',constraint3(sol.x))
            print ('constraint4 is ',constraint4(sol.x))
            print ('constraint5 is ',constraint5(sol.x))
            print ('prob_constraint is ',prob_constraint(sol.x))


        fus_PC_05 = inv(inv(C_a) + inv(C_b) - inv(C_c_05))
        C_c_EI = mutual_covariance(C_a, C_b) + 1e-10*np.identity(2)
        fus_EI = inv(inv(C_a) + inv(C_b) - inv(C_c_EI))
        
        ei = calculate_MSE(t_x_fus, C_a, C_b, C_c_EI, x_a, x_b, fus_EI)
        
        pc_05 = calculate_MSE(t_x_fus, C_a, C_b, C_c_05, x_a, x_b, fus_PC_05)
        
        
        eta = get_critical_value(2, 0.01)
        print('eta2 is ',eta)
        #Run the optimization for a different user-selected threshold
        sol = minimize(objective, S_0, method='trust-constr', constraints=cons)
        S = sol.x.reshape(2, 2).T
        C_c_10 = inv(S.T) @ inv(S)
        fus_PC_10 = inv(inv(C_a) + inv(C_b) - inv(C_c_10))
        pc_10 = calculate_MSE(t_x_fus, C_a, C_b, C_c_10, x_a, x_b, fus_PC_10)

        if(debug):
            print ('for .05, after optimization we have...')
            print('C_c_10',C_c_10)
            print ('objective is',objective(sol.x))
            print ('constraint1 is ',constraint1(sol.x))
            print ('constraint2 is ',constraint2(sol.x))
            print ('constraint3 is ',constraint3(sol.x))
            print ('constraint4 is ',constraint4(sol.x))
            print ('constraint5 is ',constraint5(sol.x))
            print ('prob_constraint is ',prob_constraint(sol.x))


        ax = plt.axes()
        plot_ellipse(fus_EI, ax, alpha_val=1, color_def="orange", linestyle="dashed")
        plot_ellipse(fus_PC_10, ax, alpha_val=.5, color_def="blue", linestyle="dashed")
        plot_ellipse(fus_PC_05, ax, alpha_val=.5, color_def="red", linestyle="dashed")
        ax.legend(['Optimal','PC=.01','PC=.05'])
        plt.show()
        
        #Store results for later analysis
        EI.append(LA.det(fus_EI))
        ei_mse.append(ei)
        PC_05.append(LA.det(fus_PC_05))
        PC_05_mse.append(pc_05)
        PC_10.append(LA.det(fus_PC_10))
        PC_10_mse.append(pc_10)
    
    print("DF:", df)    
    print("OURS .05 determinant:", sum(PC_05)/len(PC_05))
    print("OURS .01 determinant:", sum(PC_10)/len(PC_10))
    print("EI determinant:", sum(EI)/len(EI))
    print("===============================")
    print("OURS .05 MSE:", sum(PC_05_mse)/len(PC_05_mse))
    print("OURS .01 MSE:", sum(PC_10_mse)/len(PC_10_mse))
    print("EI MSE:", sum(ei_mse)/len(ei_mse))
    return (sum(PC_05)/len(PC_05), sum(PC_10)/len(PC_10), sum(EI)/len(EI)), (sum(PC_05_mse)/len(PC_05_mse), sum(PC_10_mse)/len(PC_10_mse), sum(ei_mse)/len(ei_mse))

df_s = [100]
for df in df_s:
    det, mse = run_sim(1, df)
import numpy as np
import numpy.linalg as LA


def generate_covariance(true_mu, dims):
    S = np.tril(np.random.randn(dims, dims))
    cov = np.dot(S, S.T)
    while(np.linalg.det(cov) < 0.5):
        cov = cov * 2
    mu = np.random.multivariate_normal(true_mu, cov, 1)[0]

    return mu, cov

def get(dims):
    true_mu = np.zeros((dims, ))

    x_ac, C_ac = generate_covariance(true_mu, dims)
    x_c, C_c = generate_covariance(true_mu, dims)
    x_bc, C_bc = generate_covariance(true_mu, dims)

    C_a = LA.inv(LA.inv(C_ac) + LA.inv(C_c))
    C_b = LA.inv(LA.inv(C_bc) + LA.inv(C_c))

    x_a = C_a @ (LA.inv(C_ac) @ x_ac + LA.inv(C_c) @ x_c)
    x_b = C_b @ (LA.inv(C_bc) @ x_bc + LA.inv(C_c) @ x_c)

    C_fus = LA.inv(LA.inv(C_a) + LA.inv(C_b) - LA.inv(C_c))
    print(C_fus)

    return x_a.reshape(1, dims), x_b.reshape(1, dims), C_a, C_b, C_fus
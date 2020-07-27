import numpy as np
import numpy.linalg as LA
from fusion.PC_ellipsoidal import fusion
from tqdm import tqdm

def generate_covariance(true_mu, dims):
    S = np.tril(np.random.randn(dims, dims))
    cov = np.dot(S, S.T)
    while(np.linalg.det(cov) < 0.5):
        cov = cov * 2
    mu = np.random.multivariate_normal(true_mu, cov, 1)[0]

    return mu, cov

c = 0

MSE_EI = []
det_EI = []
dist_EI = []
MSE_PC = []
det_PC = []
dist_PC = []
MSE_PC_10 = []
det_PC_10 = []
dist_PC_10 = []

count = 0
r = 0

pbar = tqdm(total= 1)

while c < 1:
    dims = 2
    true_mu = np.zeros((dims, ))

    x_ac, C_ac = generate_covariance(true_mu, dims)
    x_c, C_c = generate_covariance(true_mu, dims)
    x_bc, C_bc = generate_covariance(true_mu, dims)

    C_fus = LA.inv(LA.inv(C_ac) + LA.inv(C_bc) + LA.inv(C_c))
    x_fus = C_fus@(LA.inv(C_ac) @ x_ac + LA.inv(C_bc) @ x_bc + LA.inv(C_c) @ x_c)

    C_a = LA.inv(LA.inv(C_ac) + LA.inv(C_c))
    C_b = LA.inv(LA.inv(C_bc) + LA.inv(C_c))

    x_a = C_a @ (LA.inv(C_ac) @ x_ac + LA.inv(C_c) @ x_c)
    x_b = C_b @ (LA.inv(C_bc) @ x_bc + LA.inv(C_c) @ x_c)




    ei, eid, dist_ei, pc, pcd, dist_pc, d, f = fusion(C_a, C_b, x_a, x_b, C_fus, x_fus, C_c)
    if d:
        count += 1
    if f:
        r += 1


    MSE_EI.append(ei)
    MSE_PC.append(pc)
    dist_EI.append(dist_ei)
    det_EI.append(eid)
    det_PC.append(pcd)
    dist_PC.append(dist_pc)
    c += 1
    pbar.update(1)

pbar.close()
print("MSE: " + str(sum(MSE_EI)/len(MSE_EI)))
print("DETERMINANT: " + str(sum(det_EI)/len(det_EI)))
print("DISTANCE: " + str(sum(dist_EI)/len(dist_EI)))
print("=======================")
print("MSE: " + str(sum(MSE_PC)/len(MSE_PC)))
print("DETERMINANT: " + str(sum(det_PC)/len(det_PC)))
print("DISTANCE: " + str(sum(dist_PC)/len(dist_PC)))
print("=======================")

print(count)
print(r)
from scipy.linalg import sqrtm
from numpy.linalg import det
import numpy.linalg as LA
import torch
from scipy.stats import chi2
import numpy as np
from tqdm import tqdm

def tinv(mat):
    return torch.inverse(mat)

def normalize(arr, numpy_as = False):
    if(numpy_as):
        return arr/np.linalg.norm(arr)    
    return arr/torch.norm(arr)

def get_rotation_matrix(theta):
    dimensions = theta.shape[0] + 1
    t = torch.cat((theta, torch.zeros((4 - dimensions, 1))))
    return tgm.angle_axis_to_rotation_matrix(t.T)[:dimensions, :dimensions, :dimensions][0]
    
    
def update(z, og_Cc, C_a, C_b, x_a, x_b, zs):
    C_c = og_Cc + z.T @ z
    C_ac = torch.inverse(torch.inverse(C_a) - torch.inverse(C_c))
    C_bc = torch.inverse(torch.inverse(C_b) - torch.inverse(C_c))
    x_c = tinv(tinv(C_ac) + tinv(C_bc)) @ (tinv(C_ac) @ x_a + tinv(C_bc) @ x_b)
    x_ac = C_ac @ (torch.inverse(C_a) @ x_a - torch.inverse(C_c) @ x_c)
    x_bc = C_bc @ (torch.inverse(C_b) @ x_b - torch.inverse(C_c) @ x_c)
    Z = (x_ac - x_bc).T @ tinv(C_ac + C_bc) @ (x_ac - x_bc)
    z.retain_grad()
    Z.backward(retain_graph=True)
    zs.append(Z.data)
    return z.grad

def get_input_values(data):
    return data.get_x_a(tensor=True), data.get_x_b(tensor=True), data.get_C_a(tensor=True), data.get_C_b(tensor=True), data.get_C_c(tensor=True)


def find_direction(data):
    all_zs = []
    x_a, x_b, C_a, C_b, C_c = get_input_values(data)
    dims = x_a.size()[0]
    z = normalize(x_a - x_b).reshape((1, dims))
    z.requires_grad = True
    continue_o = True
    i = 0
    count = 0
    for i in range(100):
        count += 1
        g = update(z, C_c, C_a, C_b, x_a, x_b, all_zs)
        if len(all_zs) > 1 and all_zs[-2] <= all_zs[-1]:
            break
        z = normalize(z - g)

    return z.detach().numpy()
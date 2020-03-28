import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm

np.seterr(divide='ignore', invalid='ignore')

def compute_KL(P, Q, input):
    p = []
    q = []
    KL = 0
    dims = len(input)
    if(dims >= 3):
        X = input[0]
        Y = input[1]
        Z = input[2]
        for x in X:
            for y in Y:
                for z in Z:
                    p.append(round(P.pdf([x, y, z]), 100))
                    q.append(round(Q.pdf([x, y, z]), 100))
    else:
        X = input[0]
        Y = input[1]
        for x in X:
            for y in Y:
                p.append(round(P.pdf([x, y]), 100))
                q.append(round(Q.pdf([x, y]), 100))


    old_err_state = np.seterr(divide='raise')
    ignored_states = np.seterr(**old_err_state)
    p = np.array(p)
    q = np.array(q)
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

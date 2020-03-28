import matplotlib.pyplot as plt
import numpy as np
import math

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
    div = np.divide(p, q)
    fixed_div = []
    fixed_p = []
    for j in range(len(div)):
        i = div[j]
        if not(i == float('-inf') or i == float('+inf') or math.isnan(i) or abs(i) < 1e-200):
            fixed_div.append(i)
            fixed_p.append(p[j])
    fixed_div = np.array(fixed_div)
    fixed_p = np.array(fixed_p)
    return np.sum(fixed_p*np.log(fixed_div))


import matplotlib.pyplot as plt
import numpy as np
import math

np.seterr(divide='ignore', invalid='ignore')

def compute_KL(P, Q, X, Y):
    p = []
    q = []
    KL = 0
    for x in X:
        for y in Y:
            p.append(round(P.pdf([x, y])/100.0, 100))
            q.append(round(Q.pdf([x, y])/100.0, 100))

    old_err_state = np.seterr(divide='raise')
    ignored_states = np.seterr(**old_err_state)
    p = np.array(p)
    q = np.array(q)
    div = np.divide(p, q)
    div = np.where(div != float('+inf'), div, 0)
    div = np.where(div != float('-inf'), div, 0)
    return np.sum(np.where(div != 0, p * np.log(div), 0))


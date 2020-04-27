import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
import math
import random

def compute_independent_means(x_a, x_b, c_a, c_b, x_c, A, B, p_thresh=0.1):
    w = 0.5
    max_c_c = w*c_a + (1-w)*c_b
    c_cs = np.linspace(0, max_c_c)[0:-1]
    x_i_a = []
    x_i_b = []

    print(x_c)
    for c_c in c_cs:
        x_i_a.append((x_a/c_a - x_c/c_c)/(c_a-c_c))
        x_i_b.append((x_b/c_b - x_c/c_c)/(c_b-c_c))
    ax = plt.axes()
    ax.plot(c_cs, x_i_a, label="x_i_a", color='orange')
    ax.plot(c_cs, x_i_b, label="x_i_b", color='blue')
    plt.legend(loc='upper left', borderaxespad=0.)
    plt.ylabel("Calculated independent mean")
    plt.xlabel("Mutual Covariance")
    plt.grid(b = True)
    plt.show()
    ax = plt.axes()
    ax.plot(x_i_a, A.pdf(x_i_a), label="probability of mean A", color='orange')
    ax.plot(x_i_b, B.pdf(x_i_b), label="probability of mean B", color='blue')
    plt.legend(loc='upper left', borderaxespad=0.)
    plt.ylabel("Probability independent mean")
    plt.xlabel("Calculated Independent mean")
    plt.grid(b = True)
    plt.show()
    ax = plt.axes()
    ax.plot(c_cs, A.pdf(x_i_a), label="x_i_a", color='orange')
    ax.plot(c_cs, B.pdf(x_i_b), label="x_i_b", color='blue')
    plt.legend(loc='upper left', borderaxespad=0.)
    plt.ylabel("Probability of calculated independent mean")
    plt.xlabel("Mutual Covariance")
    ax.plot(c_cs, [p_thresh]*len(c_cs), label="P Threshold", color='red')
    plt.grid(b = True)
    plt.show()




x_a = -1
c_a = 1

x_b = 1
c_b = 1

c_c = (c_a + c_b)/2.0 - 0.01

C_bc = math.sqrt(c_b + c_c)
C_ac = math.sqrt(c_a + c_c) 
x_i_a = 0
x_i_b = 0

if not c_a == c_b:
    x_c = ((x_a*C_bc + x_b*C_ac)*(C_ac-C_bc))/(c_a-c_b)
    x_i_a = (x_a/c_a - x_c/c_c)/(c_a-c_c)
    x_i_b = (x_b/c_b - x_c/c_c)/(c_b-c_c)
else:
    x_c = (x_a + x_b)/2.0

A = multivariate_normal(x_a, c_a)
B = multivariate_normal(x_b, c_b)
# C = multivariate_normal(x_c, c_c)
# D = multivariate_normal(x_i_a, c_a)
# E = multivariate_normal(x_i_b, c_b)

# min_val = min(x_a-3*c_a, x_b-3*c_b)
# max_val = max(x_a+3*c_a, x_b+3*c_b)
# x = np.linspace(min_val, max_val)

# ax = plt.axes()
# y_a = A.pdf(x)
# y_b = B.pdf(x)
# y_c = C.pdf(x)
# y_d = D.pdf(x)
# y_e = E.pdf(x)
# plt.axvline(x=x_c, label="Mutual mean")

# ax.plot(x, y_a, label="A")
# ax.plot(x, y_b, label="B")
# ax.plot(x, y_c, label="C")
# ax.plot(x, y_d, label="A Ind")
# ax.plot(x, y_e, label="B ind")
# plt.legend(loc='upper left', borderaxespad=0.)
# plt.grid(b = True)
# plt.show()

compute_independent_means(x_a, x_b, c_a, c_b, x_c, A, B)


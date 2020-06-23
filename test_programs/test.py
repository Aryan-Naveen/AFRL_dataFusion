from from_scratch import solve_QPQC
import numpy as np

def test_quadratic_programming():
    print("A simple test direct qcqp...")
    z = np.array([1.,0.])
    P = np.array([[2,0.],[0.,-1]])
    q = np.array([1.,0])
    r = 0
    solve_QPQC(z,P,q,r)


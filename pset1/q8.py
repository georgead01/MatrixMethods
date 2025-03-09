import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu_factor



def LU_decomp(A):
    '''
    performs an LU decomposition of A

    attributes:

    - A: a matrix (n x n) for decomposition

    returns:
    - L: upper triangular matrix L such that LU = A
    - U: upper triangular matrix U such that LU = A
    '''
    n, m = A.shape

    assert n == m, 'A must be square'

    U = A.copy()
    L = np.eye(n)

    for k in range(n):
        L_k = np.eye(n)
        for i in range(k+1, n):
            l = U[i, k]/U[k, k]
            L[i, k] = l
            L_k[i, k] = -l

        U = L_k @ U

    return L, U


A = np.random.random((4, 4))
L, U = LU_decomp(A)


print(f'A: {A}')
print(f'L: {L}\nU: {U}')
print(f'LU = {L @ U}')

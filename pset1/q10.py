import numpy as np

X = np.array([
    [1, -1],
    [1,  1]
])/np.sqrt(2)

X_inv = np.linalg.inv(X)

A_1 = X @ np.array([
    [1.1, 0],
    [0  , 1]
]) @ X_inv

v_1 = X @ np.array([
    [1],
    [0]
])

A_2 = X @ np.array([
    [1.01, 0],
    [0   , 1]
]) @ X_inv

v_2 = X @ np.array([
    [1],
    [0]
])

print(f'A1 = {A_1}')
print(f'A2 = {A_2}')

def power_method(A, v):
    '''
    power method implementation, initialized to [1 0]^T, terminates when min{||x-v||, ||x+v||} < 10^-6

    attributes:

    - A: matrix (2 x 2)
    - v: eigenvector (2 x 1) of A (2 x 2) corresponding to the largest (absolute value) eigenvalue, must be of length 1

    returns:

    - iters: number of iteration until termination
    '''

    iters = 0

    x = np.array([
        [1],
        [0]
    ])

    while min(np.linalg.norm(x-v), np.linalg.norm(x+v)) >= 10**-6:
        x = (A @ x)/np.linalg.norm(A @ x)
        iters += 1

    return iters

print(f'iters for A1: {power_method(A_1, v_1)}')
print(f'iters for A2: {power_method(A_2, v_2)}')
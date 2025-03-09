import numpy as np

def linear_solver(U, b):

    '''
    a function that solves for x in Ux = b

    attributes:

    - U: n x n numpy array representing an upper triangular matrix
    - b: n x 1 numpy array representing a column vector

    returns:

    - x: n x 1 numpy array representing the solution for Ux = b
    '''
    
    n, m = U.shape
    assert n == m, f'U ({n} x {m}) not a square matrix'
    
    n_b, m_b = b.shape
    assert m_b == 1, f'b ({n_b} x {m_b}) not a column vector'
    assert n_b == n, f'column vector b ({n_b} x 1) must be of length {n}'

    for i in range(n):
        for j in range(i):
            assert U[i, j] == 0, 'U must be lower triangular'

    x = np.zeros((n, 1))

    for i in range(n-1, -1, -1):

        x[i] = (b[i] - U[i, :] @ x)/U[i, i]

    return x

U = np.array([
    [1, 3, 0],
    [0, 2, -1],
    [0, 0, -2]
])

b = np.array([
    [4],
    [3],
    [2]
])

print(f'NumPy Solution: {np.linalg.solve(U, b)}')
print(f'My Solution: {linear_solver(U, b)}')

epsilon = 10**-12
for _ in range(5):
    n = np.random.randint(2, 10)

    U = np.triu(np.random.random((n, n)))
    b = np.random.random((n, 1))

    np_sol = np.linalg.solve(U, b)
    my_sol = linear_solver(U, b)

    print(f'NumPy Solution: {np_sol}')
    print(f'My Solution: {my_sol}')

    assert ((np_sol - my_sol) < epsilon).all()
import numpy as np
import matplotlib.pyplot as plt

### pset1 q7 ###

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

    epsilon = 10**-6

    for i in range(n):
        for j in range(i):
            assert U[i, j] < epsilon, 'U must be upper triangular'

    x = np.zeros((n, 1))

    for i in range(n-1, -1, -1):

        x[i] = (b[i] - U[i, :] @ x)/U[i, i]

    return x

### pset1 q9 ###

def householder_transform(X, u):
    '''
    performs a housholder reflection (X w.r.t to u)

    attributes:

    - X: an (m x n) matrix to apply the reflection to (m >= n)
    - u: an (m x 1) vector to reflect the matrix along

    returns:

    - out: transformed X
    '''
    m, n = X.shape
    a, b = u.shape

    assert m >= n, f'm must be >= n, {m} is not >= {n}'
    assert a == m and b == 1, f'u must be a column vector of shape ({m} x {1}), not ({a} x {b})'

    result = u.T @ X
    result = u @ result
    return X - 2 * result

def householder(u):
    '''
    computes the householder matrix corresponding to u

    attributes:

    - u: a vector (m x 1) to reflect along

    returns:

    - H: corresponding household matrix
    '''
    m, _ = u.shape
    return  np.eye(m) - 2*u@u.T

def QR_Householder(A):
    '''
    performs a QR decomposition of A

    attributes:

    - A: a matrix (m x n) for decomposition (m >= n)

    returns:
    - Q: orthogonal matrix Q such that QR = A
    - R: upper triangular matrix R such that QR = A
    '''

    m, n = A.shape

    assert m >= n, f'm must be >= n, {m} is not >= {n}'

    Q = np.eye(m)
    R = A.copy()

    for i in range(n): 
        a_i = R[i:, i:i+1]

        assert len(a_i) == m-i

        X_21 = R[i:, :i]
        X_22 = R[i:, i:]

        e_i = np.zeros((m-i, 1))
        e_i[0] = 1
        u_i = np.linalg.norm(a_i)*e_i - a_i
        u_i /= np.linalg.norm(u_i)
        
        H = np.eye(m) 
        H[i:, i:] = householder(u_i)
        Q = Q @ H.T   

        R[i:, :i] = householder_transform(X_21, u_i)
        R[i:, i:] = householder_transform(X_22, u_i)

    return Q[:, :n], R[:n, :]

### pset2 q9 ###

t = np.arange(-2, 3)

A = np.zeros((len(t), 3))
A[:, 0] = 1
A[:, 1] = t
A[:, 2] = t**2

b = np.array([[1.9], [0.2], [-0.05], [1.9], [6.15]])

Q, R = QR_Householder(A)
x = linear_solver(R, Q.T@b).squeeze()

print(x)

plt.title('data vs. least squared best quadratic fit')
plt.plot(t, b.squeeze(), 'ro', label = 'data')
t_cont = np.linspace(-2, 3, 100)
plt.plot(t_cont, x[0]+x[1]*t_cont+x[2]*t_cont**2, '--', label = f'{x[0]:.3f}+{x[1]:.3f}t+{x[2]:.3f}t^2')
plt.legend()
plt.savefig('pset2/output/q9.png')
plt.show()
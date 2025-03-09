import numpy as np

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

A = np.random.random((4, 3))*10
Q, R = QR_Householder(A)

print(f'A: {A}')
print(f'Q: {Q}\nR: {R}')
print(f'QR: {Q @ R}')
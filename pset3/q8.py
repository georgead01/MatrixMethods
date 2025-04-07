import numpy as np

def conjugate_gradient(A, b, x_0, k_max, epsilon, return_k = False):
    '''
    Finds the solution to Ax = b using conjugate gradient method.

    Input:
    - A: a symmetric positive definite matrix (n x n).
    - b: a column vector (n x 1).
    - x_0: the initial guess, column vector (n x 1).
    - k_max: max number of iterations before termination, must be int.
    - epsilon: error tolerance, must be positive.

    Returns:
    - x: conjugate gradient solution for Ax = b given the specified params.
    '''

    x = x_0
    r = b - A @ x
    p = r
    err = np.linalg.norm(r)

    k = 0

    while k < k_max and err >= epsilon:

        alpha = (r.T @ p)/(p.T @ A @ p)
        x = x + alpha * p

        r = b - A @ x
        err = np.linalg.norm(r)

        beta = -(r.T @ p)/(p.T @ A @ p)
        p = r + beta * p

        k += 1

    if return_k:
        return x, k

    return x

if __name__ == '__main__':

    A = np.array([[3, 1, 1],
                [1, 2, 0],
                [1, 0, 3]])

    b = np.array([[5],
                [3],
                [4]])

    x_0 = np.zeros((3, 1))

    k_max = 10 ** 4
    epsilon = 10 ** -6

    print(conjugate_gradient(A, b, x_0, k_max, epsilon))
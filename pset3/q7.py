import numpy as np
import matplotlib.pyplot as plt

def steepest_descent(A, b, x_0, k_max, epsilon, return_k = False):
    '''
    Finds the solution to Ax = b using steepest descent method.

    Input:
    - A: a symmetric positive definite matrix (n x n).
    - b: a column vector (n x 1).
    - x_0: the initial guess, column vector (n x 1).
    - k_max: max number of iterations before termination, must be int.
    - epsilon: error tolerance, must be positive.

    Returns:
    - x: steepest descent solution for Ax = b given the specified params.
    '''

    x = x_0
    r = b - A @ x
    err = np.linalg.norm(r)

    k = 0

    while k < k_max and err >= epsilon:

        alpha = (r.T @ r)/(r.T @ A @ r)
        x = x + alpha * r

        r = b - A @ x
        err = np.linalg.norm(r)

        k += 1

    if return_k:
        return x, k
    return x

def LS_normal_equ(A, b, x_0, max_k, epsilon):
    '''
    Minimizes error (||Ax-b||^2)/2 by solving A.T @ A @ x = A.T @ b using steepest descent method.

    Input:
    - A: a matrix (m x n).
    - b: a column vector (m x 1).
    - x_0: the initial guess, column vector (n x 1).
    - k_max: max number of iterations before termination, must be int.
    - epsilon: error tolerance, must be positive.

    Returns:
    - x: steepest descent solution to the error optimization problem.
    '''

    return steepest_descent(A.T @ A, A.T @ b, x_0, max_k, epsilon)

if __name__ == '__main__':

    t = np.arange(-2, 3)

    A = np.zeros((len(t), 3))
    A[:, 0] = 1
    A[:, 1] = t
    A[:, 2] = t**2

    b = np.array([[1.9], [0.2], [-0.05], [1.9], [6.15]])

    x_0 = np.zeros((3, 1))

    k_max = 10 ** 4
    epsilon = 10 ** -6

    x = LS_normal_equ(A, b, x_0, k_max, epsilon).squeeze()

    print(x)

    plt.title('Data vs. LS Normal Equation Solution (Steepest Descent)')
    plt.plot(t, b.squeeze(), 'ro', label = 'data')
    t_cont = np.linspace(-2, 3, 100)
    plt.plot(t_cont, x[0]+x[1]*t_cont+x[2]*t_cont**2, '--', label = f'{x[0]:.3f}+{x[1]:.3f}t+{x[2]:.3f}t^2')
    plt.legend()
    plt.savefig('pset3/figs/q7.png')
    plt.show()
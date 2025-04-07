import numpy as np
import time
from q7 import steepest_descent
from q8 import conjugate_gradient

n = 200
alphas = [2, 3, 4]

for alpha in alphas:

    A_diag = np.random.uniform(10**-alpha, 1, n)
    A_diag[0] = 1
    A_diag[-1] = 10**-alpha

    A = np.diag(A_diag)

    b = np.random.uniform(-1, 1, (n, 1))

    x_0 = np.zeros((n, 1))

    k_max = 10**6
    epsilon = 10**-6

    start_time = time.time()
    x_sd, k_sd = steepest_descent(A, b, x_0, k_max, epsilon, True)
    sd_time = time.time()-start_time

    start_time = time.time()
    x_cg, k_cg = conjugate_gradient(A, b, x_0, k_max, epsilon, True)
    cg_time = time.time()-start_time

    print(f'alpha: {alpha}')
    print(20*'_')
    print(f'condition number: {10**alpha}')
    print(20*'_')
    print(f'SD error: {np.linalg.norm(A@x_sd-b)}')
    print(f'SD error < epsilon: {np.linalg.norm(A@x_sd-b) < epsilon}')
    print(20*'_')
    print(f'SD k: {k_sd}')
    print(f'SD time: {sd_time} sec')
    print(20*'_')
    print(f'CG error: {np.linalg.norm(A@x_cg-b)}')
    print(f'CG error < epsilon: {np.linalg.norm(A@x_cg-b) < epsilon}')
    print(20*'_')
    print(f'CG k: {k_cg}')
    print(f'CG time: {cg_time} sec')
    print(20*'_')
    print(20*'_')
import numpy as np
import matplotlib.pyplot as plt
import time

n = 10000

A = np.zeros((n, n))
for j in range(n):
    A[:, j] = np.random.uniform(0, j, n)

# uncomment to plot A
plt.title("A; A[i, j] ~ U[0, j]")
plt.imshow(A)
plt.savefig('pset3/figs/A.png')
plt.show()

B = np.random.rand(n, n)

# uncomment to plot B
plt.title("B; B[i, j] ~ U[0, 1]")
plt.imshow(B)
plt.savefig('pset3/figs/B.png')
plt.show()

# compute norms of columns of A
A_norm = np.linalg.norm(A, axis = 0)
# compute norms of columns of B
B_norm = np.linalg.norm(B, axis = 1)
# compute unnormalized p
p = A_norm * B_norm
# normalize p
p /= p.sum()

plt.title('index sampling distribution p')
plt.plot(p)
plt.savefig('pset3/figs/p_dist.png')
plt.show()

def randomized_matrix_multiplication(A, B, c):
    '''
    Approximate the product A@B using randomized matrix multiplication algorithm.

    Attributes:
    - A: the left matrix of the multiplication, must be a square matrix
    - B: the right matrix of the multiplication, must have same dimensions as A
    - c: number of sampled indices.

    Returns: An approximation of AB.
    '''
    
    n, p_a = A.shape
    p_b, m = B.shape

    # check shapes, A and B must be squares and of the same shape
    assert n == p_a, f'A must be a square matrix, instead got A of shape {n}x{p_a}'
    assert p_b == m, f'B must be a square matrix, instead got B of shape {p_b}x{m}'
    assert p_a == p_b, f'A and B dimensions must match, instead got A of shape {n}x{p_a} and B of shape {p_b}x{m}'

    # compute norms of columns of A
    A_norm = np.linalg.norm(A, axis = 0)
    # compute norms of columns of B
    B_norm = np.linalg.norm(B, axis = 1)
    # compute unnormalized p
    p = A_norm * B_norm
    # normalize p
    p /= p.sum()

    # check distribution is valid
    assert (p >= 0).all(), f'distribution must be positive, instead got negative values at {np.nonzero(p < 0)}'
    assert (abs(1-p.sum()) < 10 ** -6), f'distribution must add up to 1, instead it adds up to {p.sum()}'

    # initialize result matrix
    result = np.zeros((n, n))

    # sample c indices
    for _ in range(c):
        # sample index from distribution p
        j = np.random.choice(n, p = p)

        # get A column and B row, keep dimensions
        a = A[:, j:j+1]
        b = B[j:j+1, :]

        # add scaled outer product of A column and B row to result
        result += (a @ b)/(p[j]*c)

    return result

times = []
errs = []

start_time = time.time()
target = A@B
target_time = time.time()-start_time

target_norm = np.linalg.norm(target)

c_vals = [20, 100, 500]

for c in c_vals:
    start_time = time.time()
    output = randomized_matrix_multiplication(A, B, c)
    exec_time = time.time()-start_time

    rel_err = np.linalg.norm(output-target)/target_norm

    times.append(exec_time)
    errs.append(rel_err)

    print(20*'_')
    print(f'c: {c}\nrelative error: {rel_err}\nexecution time: {exec_time} sec')
    print(20*'_')

plt.title('execution time vs. c')
plt.xlabel('c')
plt.ylabel('execution time (sec)')
plt.plot(c_vals, times)
plt.savefig('pset3/figs/time_vs_c.png')
plt.show()

plt.title('relative error vs. c')
plt.xlabel('c')
plt.ylabel('relative error')
plt.plot(c_vals, errs)
plt.savefig('pset3/figs/err_vs_c.png')
plt.show()
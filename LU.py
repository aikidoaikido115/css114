import numpy as np

A = np.array([[2, 1, -1],
              [1, 2, 2],
              [1, 1, 1]])

b = np.array([[10], [8], [4]])

n = A.shape[0]
L = np.eye(n)
U = np.zeros((n, n))

# Perform LU factorization using Gaussian elimination
for k in range(n):
    # Compute U[k, j]
    for j in range(k, n):
        U[k, j] = A[k, j] - np.dot(L[k, :k], U[:k, j])
    # Compute L[i, k]
    for i in range(k+1, n):
        L[i, k] = (A[i, k] - np.dot(L[i, :k], U[:k, k])) / U[k, k]

# Solve Ly = Pb using forward substitution
P = np.eye(n)  # Permutation matrix (in this case, just the identity matrix)
b_permuted = P @ b
y = np.zeros((n, 1))
for i in range(n):
    y[i] = b_permuted[i] - np.dot(L[i, :i], y[:i])

# Solve Ux = y using backward substitution
x = np.zeros((n, 1))
for i in range(n-1, -1, -1):
    x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]

print(x)
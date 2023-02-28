import numpy as np
import scipy.linalg

A = np.array([[2, 1, -1],
              [1, 2, 2],
              [1, 1, 1]])

b = np.array([[10], [8], [4]])

P, L, U = scipy.linalg.lu(A)

print(P)
print(L)
print(U)

y = np.linalg.solve(L, P @ b)
x = np.linalg.solve(U, y)

print(x)

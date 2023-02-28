import numpy as np


A = np.array([[2, 1, -1],
              [1, 2, 2],
              [1, 1, 1]], dtype=np.int32)

b = np.array([[10], [8], [4]], dtype=np.float64)

def gauss_jordan_elimination(A, b):
    """
    Solves the system of linear equations Ax = b using Gauss-Jordan elimination.
    
    Parameters:
    A: numpy array, the coefficient matrix
    b: numpy array, the vector of constants
    
    Returns:
    x: numpy array, the solution vector
    """
    
    # Augment the coefficient matrix with the vector of constants
    A = np.hstack((A, b))
    
    # Iterate over rows
    n, m = A.shape
    for i in range(n):
        
        # Divide the ith row by the ith element to make it equal to 1
        factor = A[i, i]
        A[i, :] /= factor
        
        # Subtract multiples of the ith row from other rows to make their ith element equal to 0
        for j in range(n):
            if i != j:
                factor = A[j, i] / A[i, i]
                A[j, :] -= factor * A[i, :]
    
    # Extract the solution vector
    x = A[:, -1]
    
    return x.reshape(len(x),1)



print(gauss_jordan_elimination(A,b))
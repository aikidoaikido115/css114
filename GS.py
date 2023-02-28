import numpy as np

A = np.array([[2, 1, -1],
              [1, 2, 2],
              [1, 1, 1]], dtype=np.int32)

b = np.array([[10], [8], [4]], dtype=np.float64)

def gauss_elimination(A, b):
    """
    Solves the system of linear equations Ax = b using Gauss elimination.
    
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
        
        # Find row with maximum element in the ith column and swap rows
        max_row = i + np.argmax(abs(A[i:, i]))
        A[[i, max_row]] = A[[max_row, i]]
        
        # Subtract multiples of the ith row to eliminate lower elements in the ith column
        for j in range(i+1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
    
    # Back substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (A[i, -1] - A[i, i+1:n] @ x[i+1:]) / A[i, i]
    
    return x.reshape(len(x),1)



print(gauss_elimination(A,b))
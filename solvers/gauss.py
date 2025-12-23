import numpy as np

def gauss_elimination(A, b):
    """
    Solves the system Ax = b using Gaussian Elimination with partial pivoting.
    
    Args:
        A: Coefficients matrix (n x n).
        b: Constant vector (n).
        
    Returns:
        x: Solution vector.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    
    # Forward Elimination
    for i in range(n):
        # Partial Pivoting
        pivot = i
        for j in range(i+1, n):
            if abs(A[j, i]) > abs(A[pivot, i]):
                pivot = j
        
        # Swap rows
        A[[i, pivot]] = A[[pivot, i]]
        b[[i, pivot]] = b[[pivot, i]]
        
        if abs(A[i, i]) < 1e-10:
            raise ValueError("Matrix is singular or nearly singular")
            
        # Eliminate entries below pivot
        for j in range(i+1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]
            
    # Back Substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        sum_ax = sum(A[i, j] * x[j] for j in range(i+1, n))
        x[i] = (b[i] - sum_ax) / A[i, i]
        
    return x

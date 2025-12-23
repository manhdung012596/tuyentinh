import numpy as np

def lu_decomposition(A, b):
    """
    Solves the system Ax = b using LU Decomposition (Doolittle's Algorithm).
    A = L * U
    L * y = b
    U * x = y
    
    Args:
        A: Coefficients matrix (n x n).
        b: Constant vector (n).
        
    Returns:
        x: Solution vector.
        (L, U): The lower and upper triangular matrices.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    
    # Doolittle's Algorithm
    for i in range(n):
        # Upper Triangular
        for k in range(i, n):
            sum_val = sum(L[i, j] * U[j, k] for j in range(i))
            U[i, k] = A[i, k] - sum_val
            
        # Lower Triangular
        for k in range(i, n):
            if i == k:
                L[i, i] = 1
            else:
                sum_val = sum(L[k, j] * U[j, i] for j in range(i))
                L[k, i] = (A[k, i] - sum_val) / U[i, i]
                
    # Forward Substitution (L * y = b)
    y = np.zeros(n)
    for i in range(n):
        sum_val = sum(L[i, j] * y[j] for j in range(i))
        y[i] = b[i] - sum_val # Mean diagonal of L is 1
        
    # Back Substitution (U * x = y)
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        sum_val = sum(U[i, j] * x[j] for j in range(i+1, n))
        x[i] = (y[i] - sum_val) / U[i, i]
        
    return x, (L, U)

import numpy as np

def gauss_seidel_method(A, b, x0=None, tol=1e-6, max_iter=100):
    """
    Solves the system Ax = b using Gauss-Seidel Iterative Method.
    
    Args:
        A: Coefficients matrix (n x n).
        b: Constant vector (n).
        x0: Initial guess (n). If None, zeros are used.
        tol: Tolerance for convergence.
        max_iter: Maximum number of iterations.
        
    Returns:
        x: Solution vector.
        info: Dictionary with 'iterations' and 'converged' status.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    
    if x0 is None:
        x0 = np.zeros(n)
    
    x = x0.copy()
    history = [x.copy()]
    
    # Check for diagonal dominance (optional warning could be added here)
    if np.any(np.abs(np.diag(A)) < 1e-10):
         raise ValueError("Zero diagonal element detected, Gauss-Seidel method cannot proceed directly.")

    for k in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            s1 = sum(A[i, j] * x[j] for j in range(i))
            s2 = sum(A[i, j] * x_old[j] for j in range(i + 1, n))
            x[i] = (b[i] - s1 - s2) / A[i, i]
            
        history.append(x.copy())
        
        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            return x, {'iterations': k + 1, 'converged': True, 'history': history}
            
    return x, {'iterations': max_iter, 'converged': False, 'history': history}

import numpy as np

def sor_method(A, b, omega=1.25, x0=None, tol=1e-6, max_iter=100):
    """
    Solves the system Ax = b using Successive Over-Relaxation (SOR) Method.
    
    Args:
        A: Coefficients matrix (n x n).
        b: Constant vector (n).
        omega: Relaxation factor (0 < omega < 2).
        x0: Initial guess (n).
        tol: Tolerance.
        max_iter: Max iterations.
        
    Returns:
        x: Solution vector.
        info: Dictionary with 'iterations', 'converged' status, 'history'.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    
    if x0 is None:
        x0 = np.zeros(n)
    
    x = x0.copy()
    history = [x.copy()]
    
    for k in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            s1 = sum(A[i, j] * x[j] for j in range(i))
            s2 = sum(A[i, j] * x_old[j] for j in range(i + 1, n))
            
            # Gauss-Seidel part (this computes the new value x_GS)
            sigma = (b[i] - s1 - s2) / A[i, i]
            
            # SOR update
            # x_new = (1 - omega) * x_old + omega * x_GS
            x[i] = (1 - omega) * x_old[i] + omega * sigma
            
        history.append(x.copy())
        
        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            return x, {'iterations': k + 1, 'converged': True, 'history': history}
            
    return x, {'iterations': max_iter, 'converged': False, 'history': history}

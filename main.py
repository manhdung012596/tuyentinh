import matplotlib.pyplot as plt
import numpy as np
from solvers import gauss_elimination, lu_decomposition, gauss_seidel_method, sor_method
from visualizer import plot_system_3d, plot_convergence, plot_lu_matrices

def print_result(name, result, exact=None):
    print(f"\n--- {name} ---")
    if isinstance(result, tuple):
        x, info = result
        print(f"Solution: {x}")
        # Compact info print
        info_copy = info.copy()
        if 'history' in info_copy:
            del info_copy['history']
        print(f"Info: {info_copy}")
    else:
        x = result
        print(f"Solution: {x}")
    
    if exact is not None:
        error = np.linalg.norm(x - exact)
        print(f"L2 Error: {error:.2e}")

def main():
    print("Linear Equations Solver Demo")
    print("Methods: Gauss Elimination, LU Decomposition, Gauss-Seidel, SOR")
    
    # System Definition
    # 4x + y + z = 12
    # x + 5y + 2z = 13
    # x + 2y + 6z = 22
    # Solution: x=2, y=1, z=3
    
    A = [
        [4, 1, 1],
        [1, 5, 2],
        [1, 2, 6]
    ]
    b = [12, 13, 22]
    
    # Exact solution for comparison
    exact_solution = np.array([2.0, 1.0, 3.0])
    
    print("\nSystem:")
    print("4x + y + z = 12")
    print("x + 5y + 2z = 13")
    print("x + 2y + 6z = 22")
    print(f"Expected: {exact_solution}")

    # --- Direct Methods ---
    
    # 1. Gauss Elimination
    try:
        x_gauss = gauss_elimination(A, b)
        print_result("Gauss Elimination (Direct)", x_gauss, exact_solution)
        plot_system_3d(A, b, x_gauss)
    except Exception as e:
        print(f"Gauss Elimination failed: {e}")

    # 2. LU Decomposition
    try:
        x_lu, (L, U) = lu_decomposition(A, b)
        print_result("LU Decomposition (Direct)", x_lu, exact_solution)
        plot_lu_matrices(L, U, "lu_3d.png") # Keeping the name as requested, though content changes
    except Exception as e:
        print(f"LU Decomposition failed: {e}")

    # --- Iterative Methods ---
    
    gs_history = []
    sor_history = []

    # 3. Gauss-Seidel Method
    try:
        result_gs = gauss_seidel_method(A, b, tol=1e-8)
        print_result("Gauss-Seidel Method (Iterative)", result_gs, exact_solution)
        gs_history = result_gs[1]['history']
        # Defer showing plot
        plot_convergence(gs_history, exact_solution, "Gauss-Seidel", "gauss_seidel_convergence.png", show=False)
    except Exception as e:
        print(f"Gauss-Seidel Method failed: {e}")

    # 4. SOR Method
    try:
        result_sor = sor_method(A, b, omega=1.1, tol=1e-8)
        print_result("SOR Method (Iterative, omega=1.1)", result_sor, exact_solution)
        sor_history = result_sor[1]['history']
        # Defer showing plot
        plot_convergence(sor_history, exact_solution, "SOR", "sor_convergence.png", show=False)
    except Exception as e:
        print(f"SOR Method failed: {e}")
        
    print("\nDemo Completed. Check generated PNG files.")
    print("Opening remaining plots...")
    plt.show() # Shows GS and SOR plots together

if __name__ == "__main__":
    main()

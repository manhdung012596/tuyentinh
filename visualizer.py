import matplotlib.pyplot as plt
import numpy as np

def plot_system_3d(A, b, solution, output_file="gauss_3d.png"):
    """
    Plots the 3D planes of the equations and the solution point.
    Only works for 3x3 systems.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a meshgrid for x and y
    x = np.linspace(solution[0]-5, solution[0]+5, 20)
    y = np.linspace(solution[1]-5, solution[1]+5, 20)
    X, Y = np.meshgrid(x, y)
    
    colors = ['r', 'g', 'b']
    
    # Plot each plane: ax + by + cz = d => z = (d - ax - by) / c
    for i in range(3):
        if abs(A[i][2]) > 1e-6:
            Z = (b[i] - A[i][0]*X - A[i][1]*Y) / A[i][2]
            ax.plot_surface(X, Y, Z, alpha=0.3, color=colors[i], label=f'Eq {i+1}')
        else:
            # Handle cases where c is close to 0 (vertical planes) - simplified
            pass
            
    # Plot intersection point
    ax.scatter(solution[0], solution[1], solution[2], color='k', s=100, label='Solution')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D System Visualization (Gauss Elimination Result)')
    
    # Manually adding legend for surfaces is tricky in matplotlib, simplified title
    plt.savefig(output_file)
    print(f"Saved {output_file}")
    plt.show()
    plt.close()

def plot_convergence(history, exact_solution, method_name, output_file):
    """
    Plots the L2 error vs iteration for a single method.
    """
    errors = [np.linalg.norm(x - exact_solution) for x in history]
    iterations = range(len(history))
    
    plt.figure(figsize=(8, 6))
    plt.plot(iterations, errors, marker='o')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Error (L2 Norm)')
    plt.title(f'{method_name} Convergence')
    plt.grid(True, which="both", ls="-")
    
    plt.savefig(output_file)
    print(f"Saved {output_file}")
    plt.show()
    plt.close()

def plot_comparison(gs_history, sor_history, exact_solution, output_file="comparison.png"):
    """
    Plots the L2 error vs iteration for Gauss-Seidel vs SOR.
    """
    gs_errors = [np.linalg.norm(x - exact_solution) for x in gs_history]
    sor_errors = [np.linalg.norm(x - exact_solution) for x in sor_history]
    
    plt.figure(figsize=(10, 7))
    plt.plot(gs_errors, label='Gauss-Seidel', marker='x')
    plt.plot(sor_errors, label='SOR', marker='^')
    
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Error (L2 Norm)')
    plt.title('Convergence Comparison: Gauss-Seidel vs SOR')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    
    plt.savefig(output_file)
    print(f"Saved {output_file}")
    plt.show()
    plt.show()
    plt.close()

def plot_lu_matrices(L, U, output_file="lu_matrices.png"):
    """
    Plots the L and U matrices as heatmaps.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot L
    cax1 = ax1.matshow(L, cmap='Blues')
    fig.colorbar(cax1, ax=ax1)
    ax1.set_title('Lower Triangular Matrix (L)')
    for (i, j), z in np.ndenumerate(L):
        ax1.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')
        
    # Plot U
    cax2 = ax2.matshow(U, cmap='Greens')
    fig.colorbar(cax2, ax=ax2)
    ax2.set_title('Upper Triangular Matrix (U)')
    for (i, j), z in np.ndenumerate(U):
        ax2.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')
        
    plt.suptitle('LU Decomposition Visualization')
    plt.savefig(output_file)
    print(f"Saved {output_file}")
    plt.show()
    plt.close()

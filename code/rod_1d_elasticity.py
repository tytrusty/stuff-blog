"""
Examples for 1D Elastic Rod

Demonstrates various solvers for 1D elastic rods with nonlinear energy:
- Newton's method
- Gradient descent
- Coordinate descent with Schur complement
- Coordinate condensation
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from rod_1d import Rod1D


def visualize_schur_subspace_bases(rod, x=None, fixed_dofs=None, save_path=None, vertices_to_plot=None):
    """
    Visualize the Schur complement subspace bases for each DOF
    
    Parameters:
    -----------
    rod : Rod1D
        The rod object
    x : torch.Tensor or None
        Current positions (if None, uses rest positions)
    fixed_dofs : list or None
        Indices of fixed degrees of freedom
    save_path : str or None
        Path to save the figure
    vertices_to_plot : list or None
        Specific vertex indices to visualize (e.g., [1, 3, 7]). If None, plots all free vertices.
    """
    if x is None:
        x = rod.X.clone()
    
    # Compute subspace bases
    bases = rod.compute_schur_subspace_bases(x, fixed_dofs)
    
    # Filter bases if specific vertices are requested
    if vertices_to_plot is not None:
        bases = [b for b in bases if b['vertex'] in vertices_to_plot]
        if len(bases) == 0:
            print(f"Warning: No bases found for vertices {vertices_to_plot}")
            return None
    
    # Create figure
    n_bases = len(bases)
    n_cols = min(3, n_bases)
    n_rows = (n_bases + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_bases == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else axes
    
    X_np = rod.X.detach().numpy()
    
    for idx, basis_info in enumerate(bases):
        ax = axes[idx]
        i = basis_info['vertex']
        U_i = basis_info['basis'].detach().numpy()
        complement_dofs = basis_info['complement_dofs']
        
        # Create full displacement vector (including zero at vertex i)
        full_U = np.ones(len(rod.X))
        full_U[complement_dofs] = U_i
        
        # Plot the basis vector
        ax.plot(X_np, full_U, 'o-', linewidth=2, markersize=8, label=f'$U_{{{i}}}$')
        ax.axvline(x=X_np[i], color='red', linestyle='--', alpha=0.5, label=f'Vertex {i}')
        
        ax.set_xlabel('Rest Position')
        ax.set_ylabel('Displacement')
        ax.set_title(f'Basis for Vertex {i}')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Hide unused subplots
    for idx in range(n_bases, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved subspace basis visualization to {save_path}")
    
    return fig


def _print_results(rod, filename):
    """Helper to plot and print rod results"""
    fig = rod.plot(show_rest=True)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    
    F = rod.deformation_gradient(rod.x)
    print(f"\nDeformation gradients per element:")
    for i, f in enumerate(F):
        print(f"  Element {i}: F = {f.item():.4f}")
    print(f"Total elastic energy: {rod.total_energy(rod.x).item():.6f}")


def example_extension():
    """Example: Extension of a rod with fixed ends"""
    print("="*60)
    print("Example 1: Rod Extension with One End Pulled")
    print("="*60)
    
    rod = Rod1D(n_vertices=21, length=1.0, density=0.0, stiffness=10.0)
    rod.x[0] = rod.X[0]
    rod.x[-1] = rod.X[-1] + 0.3
    
    fixed_dofs = [0, rod.n_vertices - 1]
    rod.solve_coordinate_descent(fixed_dofs=fixed_dofs, max_iterations=1000, tolerance=1e-3, alpha=1.0)
    _print_results(rod, 'rod_extension.png')


def example_compression():
    """Example: Compression of a rod"""
    print("\n" + "="*60)
    print("Example 2: Rod Compression")
    print("="*60)
    
    rod = Rod1D(n_vertices=11, length=1.0, density=0.01, stiffness=100.0)
    rod.x[0] = rod.X[0]
    rod.x[-1] = rod.X[-1] - 0.2
    
    fixed_dofs = [0, rod.n_vertices - 1]
    rod.solve_newtons(fixed_dofs=fixed_dofs, max_iterations=50, tolerance=1e-8)
    _print_results(rod, 'rod_compression.png')


def example_free_equilibrium():
    """Example: Find equilibrium from perturbed state with only ends fixed"""
    print("\n" + "="*60)
    print("Example 3: Free Equilibrium from Perturbed State")
    print("="*60)
    
    rod = Rod1D(n_vertices=21, length=1.0, density=0.1, stiffness=50.0)
    rod.x[5:16] += torch.sin(torch.linspace(0, np.pi, 11)) * 0.1
    
    fixed_dofs = [0, rod.n_vertices - 1]
    rod.solve_newtons(fixed_dofs=fixed_dofs, max_iterations=100, tolerance=1e-10)
    
    fig = rod.plot(show_rest=True)
    plt.savefig('rod_free_equilibrium.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nTotal elastic energy: {rod.total_energy(rod.x).item():.6f}")


def example_schur_subspace_visualization():
    """Example: Visualize Schur complement subspace bases"""
    print("\n" + "="*60)
    print("Example 4: Schur Complement Subspace Bases Visualization")
    print("="*60)
    
    rod = Rod1D(n_vertices=11, length=1.0, density=1.0, stiffness=0.1)
    fixed_dofs = [0, rod.n_vertices - 1]
    
    print(f"\nSystem size: {rod.n_vertices} vertices, Fixed DOFs: {fixed_dofs}")
    
    visualize_schur_subspace_bases(rod, rod.X, fixed_dofs, 
                                    save_path='schur_subspace_bases_subset.png',
                                    vertices_to_plot=[1, 3, 7])
    plt.show()
    
    visualize_schur_subspace_bases(rod, rod.X, fixed_dofs,
                                    save_path='schur_subspace_bases_all.png')
    plt.show()


if __name__ == "__main__":
    # Run examples
    example_extension()
    # example_compression()
    # example_free_equilibrium()
    # example_schur_subspace_visualization()
    
    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("="*60)

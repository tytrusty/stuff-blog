"""
1D Elastic Rod with Nonlinear Energy

This module implements a 1D elastic rod with:
- Uniformly spaced polyline mesh with linear elements
- Multiple elastic energy density options (quadratic, cubic, quartic, neo-Hookean)
- Iterative solvers: Newton's method, gradient descent, coordinate descent, coordinate condensation
- PyTorch for automatic differentiation of gradients and Hessians
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


class Rod1D:
    """1D elastic rod with nonlinear elasticity"""
    
    def __init__(self, n_vertices, length=1.0, density=1.0, stiffness=1.0, energy_type='quadratic'):
        """
        Initialize 1D rod mesh
        
        Parameters:
        -----------
        n_vertices : int
            Number of vertices in the rod
        length : float
            Total length of the rod
        density : float
            Mass at each vertex for the diagonal mass matrix
        stiffness : float
            Stiffness coefficient for the elastic energy
        energy_type : str
            Type of elastic energy density:
            - 'quadratic': W = k * (F - 1)^2 (default, convex)
            - 'cubic': W = k * (F - 1)^3 (nonlinear, non-convex)
            - 'quartic': W = k * (F - 1)^4 (more nonlinear, convex)
            - 'neohookean': W = k * (F^2 - 1 - 2*log(F)) (physically-based)
        """
        self.n_vertices = n_vertices
        self.n_elements = n_vertices - 1
        self.length = length
        self.stiffness = stiffness
        self.energy_type = energy_type
        
        # Create uniform rest positions X
        self.X = torch.linspace(0, length, n_vertices, dtype=torch.float64)
        
        # Initialize deformed positions x (start at rest)
        self.x = self.X.clone()
        
        # Create diagonal mass matrix
        mass = density * length / n_vertices
        self.M = torch.ones(n_vertices, dtype=torch.float64) * mass
        
    def deformation_gradient(self, x):
        """
        Compute deformation gradient F for each element
        F_i = (x_{i+1} - x_i) / (X_{i+1} - X_i)
        
        Parameters:
        -----------
        x : torch.Tensor
            Current deformed positions (n_vertices,)
            
        Returns:
        --------
        F : torch.Tensor
            Deformation gradient for each element (n_elements,)
        """
        # Deformed edge lengths
        dx = x[1:] - x[:-1]
        
        # Rest edge lengths
        dX = self.X[1:] - self.X[:-1]
        
        # Deformation gradient
        F = dx / dX
        
        return F
    
    def elastic_energy_density(self, F):
        """
        Elastic energy density per element
        
        Parameters:
        -----------
        F : torch.Tensor
            Deformation gradient for each element
            
        Returns:
        --------
        W : torch.Tensor
            Energy density for each element
        """
        if self.energy_type == 'quadratic':
            # Quadratic energy: W = k * (F - 1)^2
            # Convex, symmetric around F=1
            strain = F - 1.0
            W = self.stiffness * strain**2
            
        elif self.energy_type == 'cubic':
            # Cubic energy: W = k * (F - 1)^3
            # Non-convex, asymmetric (different behavior for tension vs compression)
            strain = F - 1.0
            W = self.stiffness * strain**3
            
        elif self.energy_type == 'quartic':
            # Quartic energy: W = k * (F - 1)^4
            # More nonlinear than quadratic, but still convex
            strain = F - 1.0
            W = self.stiffness * strain**4
            
        elif self.energy_type == 'neohookean':
            # Neo-Hookean energy: W = k/2 * (F^2 - 1 - 2*log(F))
            # Physically-based, prevents compression to zero
            # Note: only valid for F > 0
            W = 0.5 * self.stiffness * ((F-1)**2 - 2.0 * 100 * torch.log(F + 1e-10))
            
        else:
            raise ValueError(f"Unknown energy_type: {self.energy_type}. "
                           f"Choose from: 'quadratic', 'cubic', 'quartic', 'neohookean'")
        
        return W
    
    def total_energy(self, x):
        """
        Compute total elastic energy of the rod
        
        Parameters:
        -----------
        x : torch.Tensor
            Current deformed positions
            
        Returns:
        --------
        energy : torch.Tensor (scalar)
            Total elastic energy
        """
        F = self.deformation_gradient(x)
        W = self.elastic_energy_density(F)
        
        # Integrate over elements (multiply by rest length of each element)
        dX = self.X[1:] - self.X[:-1]
        energy = torch.sum(W * dX)
        
        return energy
    
    def compute_gradient(self, x):
        """
        Compute gradient of elastic energy using autodiff
        
        Parameters:
        -----------
        x : torch.Tensor
            Current deformed positions
            
        Returns:
        --------
        g : torch.Tensor
            Gradient of energy w.r.t. positions
        """
        x_var = x.clone().detach().requires_grad_(True)
        energy = self.total_energy(x_var)
        energy.backward()
        
        return x_var.grad
    
    def compute_hessian(self, x):
        """
        Compute Hessian of elastic energy using autodiff
        
        Parameters:
        -----------
        x : torch.Tensor
            Current deformed positions
            
        Returns:
        --------
        H : torch.Tensor
            Hessian matrix (n_vertices, n_vertices)
        """
        n = len(x)
        H = torch.zeros(n, n, dtype=torch.float64)

        x_var = x.clone().detach().requires_grad_(True)
        hessian_matrix = torch.autograd.functional.hessian(self.total_energy, x_var)
        H = hessian_matrix.detach()        
        return H
    
    def _apply_boundary_conditions(self, H, g, fixed_dofs):
        """Apply boundary conditions to system matrix and gradient"""
        if fixed_dofs is not None:
            for dof in fixed_dofs:
                H[dof, :] = 0.0
                H[:, dof] = 0.0
                H[dof, dof] = 1.0
                g[dof] = 0.0
        return H, g
    
    def newton_step(self, x, fixed_dofs=None):
        """
        Perform one Newton iteration: solve (M + K) dx = -g
        
        Returns: (dx, residual_norm)
        """
        g = self.compute_gradient(x)
        K = self.compute_hessian(x)
        H = torch.diag(self.M) + K
        
        H, g = self._apply_boundary_conditions(H, g, fixed_dofs)
        residual_norm = torch.norm(g).item()
        dx = torch.linalg.solve(H, -g)
        
        return dx, residual_norm
    
    def compute_schur_subspace_bases(self, x, fixed_dofs=None, pin_dofs=True):
        """
        Compute Schur complement subspace basis for each DOF
        
        For each vertex i, we form a 2x2 block system from H = M + K:
        - H_ii: 1x1 block (diagonal entry for vertex i)
        - H_ic: 1x(n-1) block (row entries for other DOFs)
        - H_cc: (n-1)x(n-1) block (entries for other DOFs)
        
        The basis for vertex i is: U_i = -H_cc^{-1} H_ic^T
        
        Parameters:
        -----------
        x : torch.Tensor
            Current positions
        fixed_dofs : list or None
            Indices of fixed degrees of freedom (excluded from subspace)
            
        Returns:
        --------
        subspace_bases : list of torch.Tensor
            List of (n-1)x1 basis vectors, one per DOF
        """
        # Compute system matrix H = M + K
        K = self.compute_hessian(x)
        M_diag = torch.diag(self.M)
        H = M_diag + K
        
        n = len(x)
        subspace_bases = []
        
        # Determine which DOFs are free
        if fixed_dofs is None:
            fixed_dofs = []
        free_dofs = [i for i in range(n) if i not in fixed_dofs]

        # Update hessians for fixed DOFs (set 1 on diagonal and 0 otherwise)
        if pin_dofs:
            for dof in fixed_dofs:
                H[dof, :] = 0.0
                H[:, dof] = 0.0
                H[dof, dof] = 1.0
        
        for i in free_dofs:
            # Get indices for the complementary DOFs (all except i)
            complement_dofs = [j for j in range(n) if j != i]
            
            # Extract blocks
            H_ii = H[i, i].unsqueeze(0).unsqueeze(0)  # 1x1
            H_ic = H[i, complement_dofs].unsqueeze(0)  # 1x(n-1)
            H_cc = H[complement_dofs, :][:, complement_dofs]  # (n-1)x(n-1)
            
            # Compute basis: U_i = -H_cc^{-1} H_ic^T
            try:
                H_cc_inv = torch.linalg.inv(H_cc)
                U_i = -H_cc_inv @ H_ic.T  # (n-1)x1
            except:
                # If H_cc is singular, use pseudoinverse
                H_cc_inv = torch.linalg.pinv(H_cc)
                U_i = -H_cc_inv @ H_ic.T
            
            subspace_bases.append({
                'vertex': i,
                'basis': U_i.squeeze(),  # (n-1,)
                'complement_dofs': complement_dofs
            })
        
        return subspace_bases
    
    def iterative_solve(self, search_direction, solver_name, max_iterations, tolerance, fixed_dofs, verbose):
        """
        Generic iterative solver framework
        
        Iteratively computes search direction and updates position until convergence:
        while not converged:
            d, res_norm = search_direction(x, fixed_dofs)
            if res_norm < tol: break
            x = x + d
        
        Parameters:
        -----------
        search_direction : callable
            Function that computes search direction: (d, residual_norm) = search_direction(x, fixed_dofs)
        solver_name : str
            Name of the solver for verbose output
        """
        if verbose:
            print(f"Starting {solver_name} solver (tol={tolerance}, max_iter={max_iterations})")
            print(f"{'Iter':<6} {'Residual':<15} {'Step Size':<15}")
            print("-" * 40)
        
        for iteration in range(max_iterations):
            d, res_norm = search_direction(self.x, fixed_dofs)
            
            if verbose:
                d_norm = torch.norm(d).item()
                print(f"{iteration:<6} {res_norm:<15.6e} {d_norm:<15.6e}")
            
            if res_norm < tolerance:
                if verbose:
                    print(f"\nConverged in {iteration + 1} iterations!")
                return True, iteration + 1
            
            self.x = self.x + d
        
        if verbose:
            print(f"\nWarning: Did not converge in {max_iterations} iterations")
            print(f"Final residual: {res_norm:.6e}")
        
        return False, max_iterations
    
    def solve_newtons(self, max_iterations=100, tolerance=1e-6, fixed_dofs=None, verbose=True):
        """Solve quasistatic equilibrium using Newton's method"""
        return self.iterative_solve(
            self.newton_step, "Newton", max_iterations, tolerance, fixed_dofs, verbose
        )
    
    def gradient_descent_step(self, x, step_size, fixed_dofs=None):
        """Compute gradient descent search direction: d = -step_size * gradient"""
        g = self.compute_gradient(x)
        if fixed_dofs is not None:
            for dof in fixed_dofs:
                g[dof] = 0.0
        residual_norm = torch.norm(g).item()
        d = -step_size * g
        return d, residual_norm
    
    def solve_gradient_descent(self, max_iterations=100, tolerance=1e-6, step_size=0.01, fixed_dofs=None, verbose=True):
        """Solve quasistatic equilibrium using gradient descent"""
        search_direction = lambda x, fd: self.gradient_descent_step(x, step_size, fd)
        return self.iterative_solve(
            search_direction, "Gradient Descent", max_iterations, tolerance, fixed_dofs, verbose
        )
    
    def _setup_schur_step(self, x, fixed_dofs):
        """Setup common H, g for Schur step methods"""
        g = self.compute_gradient(x)
        K = self.compute_hessian(x)
        H = torch.diag(self.M) + K
        H, g = self._apply_boundary_conditions(H, g, fixed_dofs)
        residual_norm = torch.norm(g).item()
        dx = torch.zeros_like(x)
        return H, g, dx, residual_norm
    
    def schur_subspace_step(self, x, subspace_bases, alpha=1.0, fixed_dofs=None, gauss_seidel=False):
        """
        Perform one iteration of the Schur complement subspace solver
        
        For each vertex i, solve:
        dx_i = (H_ii + alpha * U_ic^T H_cc U_ic)^{-1} (g_i + alpha * U_ic^T g_c)
        """
        H, g, dx, residual_norm = self._setup_schur_step(x, fixed_dofs)
        
        # Solve for each free vertex individually
        for basis_info in subspace_bases:
            i = basis_info['vertex']
            U_ic = basis_info['basis']  # (n-1,) basis vector
            complement_dofs = basis_info['complement_dofs']
            
            # Extract blocks
            H_ii = H[i, i]  # scalar
            H_ic = H[i, complement_dofs]  # (n-1,)
            H_cc = H[complement_dofs, :][:, complement_dofs]  # (n-1, n-1)

            if gauss_seidel:
                # update the gradient
                g = self.compute_gradient(x + dx)
            
            g_i = g[i]  # scalar
            g_c = g[complement_dofs]  # (n-1,)
            
            # Compute reduced system:
            # dx_i = (H_ii + alpha * U_ic^T H_cc U_ic)^{-1} (g_i + alpha * U_ic^T g_c)
            # Left-hand side: scalar
            U_ic = U_ic.unsqueeze(1)
            lhs = H_ii + alpha * (U_ic.T @ H_cc @ U_ic)
            
            # Right-hand side: scalar
            rhs = -(g_i + alpha * (U_ic.T @ g_c))
            
            # Solve for dx_i
            dx_i = rhs / lhs
            
            # Update dx for vertex i
            dx[i] = dx_i
            
        # Apply boundary conditions (zero out fixed DOFs)
        if fixed_dofs is not None:
            for dof in fixed_dofs:
                dx[dof] = 0.0
        
        return dx, residual_norm
    
    def fixed_schur_subspace_step(self, x, subspace_bases, fixed_dofs=None):
        """
        Perform one iteration of the fixed Schur complement solver with 2x2 block solves
        
        For each vertex i, solve the 2x2 system:
        [H_ii          H_ic U_ic    ] [dx_i] = -[g_i      ]
        [(H_ic U_ic)^T  U_ic^T H_cc U_ic] [dc_i]    [U_ic^T g_c]
        """
        H, g, dx, residual_norm = self._setup_schur_step(x, fixed_dofs)
        
        # Solve for each free vertex individually with 2x2 block system
        for basis_info in subspace_bases:
            i = basis_info['vertex']
            U_ic = basis_info['basis']  # (n-1,) basis vector
            complement_dofs = basis_info['complement_dofs']
            
            # Extract blocks
            H_ii = H[i, i]  # scalar
            H_ic = H[i, complement_dofs]  # (n-1,)
            H_cc = H[complement_dofs, :][:, complement_dofs]  # (n-1, n-1)
            
            g_i = g[i]  # scalar
            g_c = g[complement_dofs]  # (n-1,)
            
            # Form 2x2 block system
            U_ic = U_ic.unsqueeze(1)  # (n-1, 1)
            
            # Block components
            # H_11 = H_ii (scalar)
            H_11 = H_ii.unsqueeze(0).unsqueeze(0)  # (1, 1)
            
            # H_12 = H_ic U_ic (1x1)
            H_12 = (H_ic.unsqueeze(0) @ U_ic)  # (1, 1)
            
            # H_21 = (H_ic U_ic)^T = U_ic^T H_ic^T (1x1)
            H_21 = H_12.T  # (1, 1)
            
            # H_22 = U_ic^T H_cc U_ic (1x1)
            H_22 = U_ic.T @ H_cc @ U_ic  # (1, 1)
            
            # Form 2x2 system matrix
            A = torch.zeros((2, 2), dtype=torch.float64)
            A[0, 0] = H_11.item()
            A[0, 1] = H_12.item()
            A[1, 0] = H_21.item()
            A[1, 1] = H_22.item()
            
            # Right-hand side
            b = torch.zeros(2, dtype=torch.float64)
            b[0] = -g_i
            b[1] = -(U_ic.T @ g_c).item()
            
            # Solve 2x2 system
            solution = torch.linalg.solve(A, b)
            dx_i = solution[0]  # Only keep the first component (dx_i)
            # dc_i = solution[1]  # We don't use this
            
            # Update dx for vertex i
            dx[i] = dx_i
            
        # Apply boundary conditions (zero out fixed DOFs)
        if fixed_dofs is not None:
            for dof in fixed_dofs:
                dx[dof] = 0.0
        
        return dx, residual_norm
    
    def solve_coordinate_descent(self, max_iterations=100, tolerance=1e-6, alpha=1.0, fixed_dofs=None, verbose=True, gauss_seidel=False):
        """Solve using coordinate descent with Schur complement subspace"""
        if verbose:
            print("Computing Schur complement subspace bases...")
        subspace_bases = self.compute_schur_subspace_bases(self.x, fixed_dofs)
        if verbose:
            print(f"Computed {len(subspace_bases)} subspace bases")
        
        search_direction = lambda x, fd: self.schur_subspace_step(
            x, subspace_bases, alpha=alpha, fixed_dofs=fd, gauss_seidel=gauss_seidel
        )
        return self.iterative_solve(
            search_direction, "Coordinate Descent", max_iterations, tolerance, fixed_dofs, verbose
        )
    
    def solve_coordinate_condensation(self, max_iterations=100, tolerance=1e-6, fixed_dofs=None, verbose=True):
        """Solve using coordinate condensation with 2x2 block solves"""
        if verbose:
            print("Computing Schur complement subspace bases...")
        subspace_bases = self.compute_schur_subspace_bases(self.x, fixed_dofs)
        if verbose:
            print(f"Computed {len(subspace_bases)} subspace bases")
        
        search_direction = lambda x, fd: self.fixed_schur_subspace_step(x, subspace_bases, fixed_dofs=fd)
        return self.iterative_solve(
            search_direction, "Coordinate Condensation", max_iterations, tolerance, fixed_dofs, verbose
        )
    
    
    def plot(self, show_rest=True):
        """
        Plot the rod configuration
        
        Parameters:
        -----------
        show_rest : bool
            Whether to show the rest configuration
        """
        plt.figure(figsize=(10, 4))
        
        # Plot deformed configuration
        x_np = self.x.detach().numpy()
        y_deformed = np.zeros_like(x_np)
        plt.plot(x_np, y_deformed, 'o-', linewidth=2, markersize=8, label='Deformed')
        
        # Plot rest configuration
        if show_rest:
            X_np = self.X.detach().numpy()
            y_rest = np.ones_like(X_np) * 0.05
            plt.plot(X_np, y_rest, 'o--', linewidth=1, markersize=6, 
                    alpha=0.5, label='Rest', color='gray')
        
        plt.xlabel('Position')
        plt.ylabel('Displacement')
        plt.title('1D Rod')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        
        return plt.gcf()


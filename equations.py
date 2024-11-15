from scipy import sparse
from timesteppers import StateVector, CrankNicolson, RK22
import numpy as np
import finite
from scipy.sparse import linalg as spla
from scipy.sparse.linalg import spsolve
from finite import DifferenceUniformGrid
from scipy.sparse import diags, kron, eye,csc_matrix, lil_matrix
from scipy.sparse.linalg import splu
from math import factorial


class DiffusionBC:
    def __init__(self, c, D, spatial_order, domain):
        self.t = 0
        self.iter = 0
        self.c = c.copy()
        self.D = D
        self.Nx, self.Ny = c.shape
        self.dx = domain.grids[0].dx
        self.dy = domain.grids[1].dx

        # Initialize matrices for the scheme
        self.I = sparse.eye(self.Nx * self.Ny)
        self.dt = None

        # Build spatial derivative operators
        self.build_operators(spatial_order)

    def build_operators(self, spatial_order):
        # Build second derivative operators for x and y directions

        # X direction (non-periodic with boundary conditions)
        main_diag = np.ones(self.Nx) * -2.0
        off_diag = np.ones(self.Nx - 1)
        diagonals = []
        offsets = []

        if spatial_order == 2:
            diagonals = [off_diag, main_diag, off_diag]
            offsets = [-1, 0, 1]
        elif spatial_order == 4:
            diagonals = [
                np.ones(self.Nx - 2) * (-1 / 12),  # -2 offset
                np.ones(self.Nx - 1) * (4 / 3),  # -1 offset
                np.ones(self.Nx) * (-5 / 2),  # main diagonal
                np.ones(self.Nx - 1) * (4 / 3),  # +1 offset
                np.ones(self.Nx - 2) * (-1 / 12)  # +2 offset
            ]
            offsets = [-2, -1, 0, 1, 2]

        dx2 = sparse.diags(diagonals, offsets, shape=(self.Nx, self.Nx), format='lil') / (self.dx ** 2)

        # Apply boundary conditions
        # Left boundary: Dirichlet (c = 0)
        dx2[0, :] = 0
        dx2[0, 0] = 1

        # Right boundary: Neumann (dc/dx = 0)
        if spatial_order == 2:
            dx2[-1, -3:] = np.array([1, -4, 3]) / (2 * self.dx)
        elif spatial_order == 4:
            dx2[-1, -5:] = np.array([-1, 8, -13, 8, -2]) / (12 * self.dx)

        # Y direction (periodic)
        diagonals_y = []
        offsets_y = []

        if spatial_order == 2:
            main_diag_y = np.ones(self.Ny) * -2.0
            off_diag_y = np.ones(self.Ny - 1)
            diagonals_y = [off_diag_y, main_diag_y, off_diag_y]
            offsets_y = [-1, 0, 1]
            dy2 = sparse.diags(diagonals_y, offsets_y, shape=(self.Ny, self.Ny), format='lil') / (self.dy ** 2)
            # Add periodic connections
            dy2[0, -1] = 1 / (self.dy ** 2)
            dy2[-1, 0] = 1 / (self.dy ** 2)

        elif spatial_order == 4:
            main_diag_y = np.ones(self.Ny) * -5 / 2
            off1_diag_y = np.ones(self.Ny - 1) * 4 / 3
            off2_diag_y = np.ones(self.Ny - 2) * -1 / 12
            diagonals_y = [off2_diag_y, off1_diag_y, main_diag_y, off1_diag_y, off2_diag_y]
            offsets_y = [-2, -1, 0, 1, 2]
            dy2 = sparse.diags(diagonals_y, offsets_y, shape=(self.Ny, self.Ny), format='lil') / (self.dy ** 2)
            # Add periodic connections
            dy2[0, -2:] = np.array([-1 / 12, 4 / 3]) / (self.dy ** 2)
            dy2[1, -1] = -1 / 12 / (self.dy ** 2)
            dy2[-2:, 0] = np.array([-1 / 12, 4 / 3]) / (self.dy ** 2)
            dy2[-1, 1] = -1 / 12 / (self.dy ** 2)

        # Build full 2D Laplacian using Kronecker products
        Ix = sparse.eye(self.Nx)
        Iy = sparse.eye(self.Ny)
        self.L = self.D * (sparse.kron(Iy, dx2.tocsr()) + sparse.kron(dy2.tocsr(), Ix))

    def step(self, dt):
        if dt != self.dt:
            # Build Crank-Nicolson matrices
            self.LHS = self.I - dt / 2 * self.L
            self.RHS = self.I + dt / 2 * self.L
            self.dt = dt
            # Prepare LU decomposition for solving the system
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec='NATURAL')

        # Reshape solution to 1D array for matrix operations
        c_flat = self.c.reshape(-1)

        # Perform Crank-Nicolson step
        c_new = self.LU.solve(self.RHS @ c_flat)

        # Update solution and time
        self.c[:] = c_new.reshape(self.Nx, self.Ny)
        self.t += dt
        self.iter += 1



class Wave2DBC:
    def __init__(self, u, v, p, spatial_order, domain):
        # Store initial conditions and parameters
        self.u = u
        self.v = v
        self.p = p
        self.spatial_order = spatial_order
        self.domain = domain

        # Get grid information
        self.nx, self.ny = u.shape
        self.dx = domain.dx
        self.dy = domain.dy

        # Set up derivative operators
        if spatial_order == 2:
            self.x_stencil = np.array([-1 / 2, 0, 1 / 2]) / self.dx
            self.y_stencil = np.array([-1 / 2, 0, 1 / 2]) / self.dy
        elif spatial_order == 4:
            self.x_stencil = np.array([1 / 12, -2 / 3, 0, 2 / 3, -1 / 12]) / self.dx
            self.y_stencil = np.array([1 / 12, -2 / 3, 0, 2 / 3, -1 / 12]) / self.dy
        else:
            raise ValueError("Only 2nd and 4th order spatial derivatives are supported")

        # Define the F function for time stepping
        def F(X):
            u, v, p = X

            # Calculate spatial derivatives
            dp_dx = self.x_derivative(p)
            dp_dy = self.y_derivative(p)
            du_dx = self.x_derivative(u)
            dv_dy = self.y_derivative(v)

            # Wave equation system
            du_dt = -dp_dx
            dv_dt = -dp_dy
            dp_dt = -(du_dx + dv_dy)

            return np.array([du_dt, dv_dt, dp_dt])

        # Define the boundary conditions function
        def BC(X):
            u, v, p = X

            # Apply x-direction boundary conditions (u = 0 at x boundaries)
            u[0, :] = 0  # Left boundary
            u[-1, :] = 0  # Right boundary

            # Apply periodic boundary conditions in y-direction
            u[:, 0] = u[:, -2]
            u[:, -1] = u[:, 1]
            v[:, 0] = v[:, -2]
            v[:, -1] = v[:, 1]
            p[:, 0] = p[:, -2]
            p[:, -1] = p[:, 1]

            return np.array([u, v, p])

        # Store F and BC functions
        self.F = F
        self.BC = BC

    def x_derivative(self, field):
        """Calculate x derivative using finite difference."""
        if self.spatial_order == 2:
            padded = np.pad(field, ((1, 1), (0, 0)), mode='edge')
            result = np.zeros_like(field)
            for i in range(3):
                result += self.x_stencil[i] * padded[i:i + self.nx, :]
        else:  # 4th order
            padded = np.pad(field, ((2, 2), (0, 0)), mode='edge')
            result = np.zeros_like(field)
            for i in range(5):
                result += self.x_stencil[i] * padded[i:i + self.nx, :]
        return result

    def y_derivative(self, field):
        """Calculate y derivative using finite difference with periodic boundary."""
        if self.spatial_order == 2:
            padded = np.pad(field, ((0, 0), (1, 1)), mode='wrap')
            result = np.zeros_like(field)
            for i in range(3):
                result += self.y_stencil[i] * padded[:, i:i + self.ny]
        else:  # 4th order
            padded = np.pad(field, ((0, 0), (2, 2)), mode='wrap')
            result = np.zeros_like(field)
            for i in range(5):
                result += self.y_stencil[i] * padded[:, i:i + self.ny]
        return result


class ReactionDiffusion2D:
    def __init__(self, c, D, dx2, dy2):
        self.c = c
        self.D = D
        self.dx2 = dx2
        self.dy2 = dy2
        self.nx, self.ny = c.shape
        self.t = 0.0
        self.iter = 0

    def reaction_step(self, dt):
        """Fourth-order Runge-Kutta for reaction term"""

        def f(c):
            return c * (1.0 - c)

        k1 = f(self.c)
        k2 = f(self.c + 0.5 * dt * k1)
        k3 = f(self.c + 0.5 * dt * k2)
        k4 = f(self.c + dt * k3)

        self.c[:] = self.c + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        np.clip(self.c, 0, 1, out=self.c)

    def diffusion_step(self, dt):
        """Modified Crank-Nicolson diffusion step with higher accuracy"""
        from scipy.sparse.linalg import spsolve
        from scipy.sparse import eye

        # Split into x and y steps for better accuracy
        # X direction
        I = eye(self.nx, format='csr')
        A_x = (I - 0.5 * dt * self.D * self.dx2.matrix).tocsr()

        temp = np.zeros_like(self.c)
        for j in range(self.ny):
            b = self.c[:, j] + 0.5 * dt * self.D * (self.dx2.matrix @ self.c[:, j])
            temp[:, j] = spsolve(A_x, b)

        # Y direction
        I = eye(self.ny, format='csr')
        A_y = (I - 0.5 * dt * self.D * self.dy2.matrix).tocsr()

        for i in range(self.nx):
            b = temp[i, :] + 0.5 * dt * self.D * (temp[i, :] @ self.dy2.matrix)
            self.c[i, :] = spsolve(A_y, b)

    def step(self, dt):
        """
        Take a timestep using symmetric Strang splitting with
        smaller substeps for high resolution
        """
        # For high resolution, take multiple smaller steps
        if self.nx >= 200:
            n_substeps = 2
            dt_sub = dt / n_substeps
            for _ in range(n_substeps):
                # Half step of diffusion
                self.diffusion_step(dt_sub / 2)

                # Full step of reaction
                self.reaction_step(dt_sub)

                # Half step of diffusion
                self.diffusion_step(dt_sub / 2)
        else:
            # Original method for lower resolutions
            self.diffusion_step(dt / 2)
            self.reaction_step(dt)
            self.diffusion_step(dt / 2)

        self.t += dt
        self.iter += 1

        return self.c

class ViscousBurgers2D:
    def __init__(self, u, v, nu, spatial_order, domain):
        """Initialize the 2D Viscous Burgers solver.

        Args:
            u, v: Initial velocity components arrays
            nu: Viscosity coefficient
            spatial_order: Order of spatial derivatives
            domain: Domain specification for the problem
        """
        self.u = u
        self.v = v
        self.nu = nu

        # Get grid information
        self.grid_x = domain.grids[0]
        self.grid_y = domain.grids[1]
        self.nx, self.ny = u.shape

        # Initialize time and iteration counter
        self.t = 0
        self.iter = 0

        # Create derivative operators using internal methods
        self.dx = self._create_difference_operator(1, spatial_order, self.grid_x, 0)
        self.dy = self._create_difference_operator(1, spatial_order, self.grid_y, 1)
        self.dx2 = self._create_difference_operator(2, spatial_order, self.grid_x, 0)
        self.dy2 = self._create_difference_operator(2, spatial_order, self.grid_y, 1)

        # Precompute sparse matrices
        self.Ix = eye(self.nx)
        self.Iy = eye(self.ny)

        # Create full 2D diffusion operator (only once)
        self.diff_op = (kron(self.dx2, self.Iy) + kron(self.Ix, self.dy2))

        # Pre-allocate arrays for RK22
        self.k1_u = np.zeros_like(u)
        self.k1_v = np.zeros_like(v)
        self.u_mid = np.zeros_like(u)
        self.v_mid = np.zeros_like(v)
        self.k2_u = np.zeros_like(u)
        self.k2_v = np.zeros_like(v)

        # Variables for diffusion step
        self.u_flat = np.zeros(self.nx * self.ny)
        self.v_flat = np.zeros(self.nx * self.ny)
        self.prev_dt = None
        self.LU = None

    def _get_stencil_and_coeffs(self, derivative_order, convergence_order):
        """Calculate finite difference stencil and coefficients."""
        stencil_size = convergence_order + derivative_order
        half_stencil = stencil_size // 2

        # Central difference stencil
        stencil = np.arange(-half_stencil, half_stencil + 1)

        # Generate coefficients using Taylor series
        A = np.vander(stencil, increasing=True).T
        b = np.zeros_like(stencil)
        b[derivative_order] = factorial(derivative_order)

        # Solve for coefficients
        coeffs = np.linalg.solve(A, b)
        return stencil, coeffs

    def _create_difference_operator(self, derivative_order, convergence_order, grid, axis):
        """Create finite difference operator matrix."""
        N = grid.N
        dx = grid.dx

        # Calculate stencil and coefficients
        stencil_size = convergence_order + derivative_order
        half_stencil = stencil_size // 2
        stencil, coeffs = self._get_stencil_and_coeffs(derivative_order, convergence_order)

        # Create sparse matrix
        matrix = lil_matrix((N, N))

        # Fill matrix with coefficients
        for i in range(N):
            for j, coeff in enumerate(coeffs):
                index = (i + stencil[j]) % N  # Periodic boundary conditions
                matrix[i, index] = coeff / (dx ** derivative_order)

        return matrix.tocsr()

    def advection_terms(self, u, v):
        """Compute advection terms for both u and v simultaneously"""
        # Compute all derivatives at once
        ux = self.dx @ u
        uy = u @ self.dy.T
        vx = self.dx @ v
        vy = v @ self.dy.T

        # Return both terms
        return -u * ux - v * uy, -u * vx - v * vy

    def diffusion_step(self, dt):
        """Combined Crank-Nicolson step for both u and v"""
        # Only recompute LU decomposition if dt changes
        if dt != self.prev_dt:
            N = self.nx * self.ny
            I = eye(N)
            LHS = I - 0.5 * dt * self.nu * self.diff_op
            RHS = I + 0.5 * dt * self.nu * self.diff_op
            self.LU = splu(LHS.tocsc())
            self.RHS = RHS
            self.prev_dt = dt

        # Solve for u
        np.copyto(self.u_flat, self.u.reshape(-1))
        u_new = self.LU.solve(self.RHS @ self.u_flat)
        self.u[:] = u_new.reshape(self.nx, self.ny)

        # Solve for v
        np.copyto(self.v_flat, self.v.reshape(-1))
        v_new = self.LU.solve(self.RHS @ self.v_flat)
        self.v[:] = v_new.reshape(self.nx, self.ny)

    def step(self, dt):
        """Take a timestep of size dt using Strang splitting."""
        # First half step of diffusion
        self.diffusion_step(dt / 2)

        # Full step of advection using RK22
        u_init = self.u
        v_init = self.v

        # First stage
        self.k1_u, self.k1_v = self.advection_terms(u_init, v_init)

        # Midpoint values
        np.multiply(dt / 2, self.k1_u, out=self.u_mid)
        np.multiply(dt / 2, self.k1_v, out=self.v_mid)
        np.add(u_init, self.u_mid, out=self.u_mid)
        np.add(v_init, self.v_mid, out=self.v_mid)

        # Second stage
        self.k2_u, self.k2_v = self.advection_terms(self.u_mid, self.v_mid)

        # Final update
        np.multiply(dt, self.k2_u, out=self.u_mid)
        np.multiply(dt, self.k2_v, out=self.v_mid)
        np.add(u_init, self.u_mid, out=self.u)
        np.add(v_init, self.v_mid, out=self.v)

        # Second half step of diffusion
        self.diffusion_step(dt / 2)

        # Update time
        self.t += dt
        self.iter += 1

        return self.u, self.v

class ViscousBurgers:

    def __init__(self, u, nu, d, d2):
        self.u = u
        self.X = StateVector([u])

        N = len(u)
        self.M = sparse.eye(N, N)
        self.L = -nu * d2.matrix

        f = lambda X: -X.data * (d @ X.data)

        self.F = f

class Wave:

    def __init__(self, u, v, d2):
        self.X = StateVector([u, v])
        N = len(u)
        I = sparse.eye(N, N)
        Z = sparse.csr_matrix((N, N))

        M00 = I
        M01 = Z
        M10 = Z
        M11 = I
        self.M = sparse.bmat([[M00, M01],
                              [M10, M11]])

        L00 = Z
        L01 = -I
        L10 = -d2.matrix
        L11 = Z
        self.L = sparse.bmat([[L00, L01],
                              [L10, L11]])

        self.F = lambda X: 0 * X.data

class SoundWave:
    def __init__(self, u, p, d, rho0, gammap0):
        # State vector [u, p]
        self.X = StateVector([u, p])
        N = len(u)

        # Create sparse identity and zero matrices
        Z = sparse.csr_matrix((N, N))
        I = sparse.eye(N, N)

        # For scalar rho0, multiply directly with identity matrix
        if np.isscalar(rho0):
            rho0_matrix = rho0 * I
        else:
            rho0_matrix = sparse.diags(rho0)

        # Mass matrix M
        self.M = sparse.bmat([
            [rho0_matrix, Z],
            [Z, I]
        ])

        # Linear operator L
        self.L = sparse.bmat([
            [Z, d.matrix],
            [gammap0 * d.matrix, Z]
        ])

        # No nonlinear terms for sound wave equation
        self.F = lambda X: 0 * X.data

class ReactionDiffusion:

    def __init__(self, c, d2, c_target, D):
        # State vector
        self.X = StateVector([c])
        N = len(c)

        # Mass matrix M: Identity
        self.M = sparse.eye(N, N)

        # Linear operator L: -D∇²
        self.L = -D * d2.matrix

        # Nonlinear function F: c(c_target - c)
        def f(X):
            return X.data * (c_target - X.data)

        self.F = f


from scipy import sparse
import timesteppers
from timesteppers import StateVector, CrankNicolson, RK22
import numpy as np
import finite
from farray import axslice
from scipy.sparse import linalg as spla
from scipy.sparse.linalg import spsolve
from finite import DifferenceUniformGrid
from scipy.sparse import diags, kron, eye,csc_matrix, lil_matrix
from scipy.sparse.linalg import splu,spsolve, cg
from math import factorial


class DiffusionBC:
    def __init__(self, c, D, spatial_order, domain):
        if not isinstance(c, np.ndarray) or c.ndim != 2:
            raise ValueError("Initial condition 'c' must be a 2D numpy array")

        if not isinstance(D, (int, float)) or D <= 0:
            raise ValueError("Diffusion coefficient 'D' must be positive")

        if not isinstance(spatial_order, int) or spatial_order < 1:
            raise ValueError("spatial_order must be a positive integer")

        self.c = c.copy()
        self.D = float(D)
        self.spatial_order = spatial_order

        self.ny, self.nx = c.shape

        if hasattr(domain, 'grids'):
            grid_x = domain.grids[0]
            grid_y = domain.grids[1]

            self.dx = grid_x.dx
            self.dy = grid_y.dx

            x, y = domain.values()
            self.xl = x[0]
            self.xr = x[-1]
            self.yb = y[0]
            self.yt = y[-1]

        elif isinstance(domain, tuple) and len(domain) == 2:
            (self.xl, self.xr), (self.yb, self.yt) = domain
            if self.xl >= self.xr or self.yb >= self.yt:
                raise ValueError("Invalid domain: ensure xl < xr and yb < yt")
            self.dx = (self.xr - self.xl) / (self.nx - 1)
            self.dy = (self.yt - self.yb) / (self.ny - 1)
        else:
            raise ValueError("domain must be either a Domain object or a tuple of ((xl, xr), (yb, yt))")

        self.t = 0.0
        self.iter = 0

        self._build_operator()

    def _build_operator(self):
        ny, nx = self.ny, self.nx
        N = nx * ny

        A = sparse.lil_matrix((N, N))

        for j in range(ny):
            for i in range(nx):
                idx = i + j * nx

                if i == 0:  # Left boundary (Dirichlet)
                    A[idx, idx] = 1.0

                elif i == nx - 1:  # Right boundary (Neumann)
                    if self.spatial_order == 2:
                        A[idx, idx - 1] = 1.0
                        A[idx, idx] = -1.0
                    else:
                        A[idx, idx - 3] = -2.0
                        A[idx, idx - 2] = 9.0
                        A[idx, idx - 1] = -18.0
                        A[idx, idx] = 11.0

                else:  # Interior points
                    dx2 = self.dx * self.dx
                    dy2 = self.dy * self.dy

                    # x derivatives
                    A[idx, idx - 1] = self.D / dx2
                    A[idx, idx] = -2.0 * self.D * (1.0 / dx2 + 1.0 / dy2)
                    A[idx, idx + 1] = self.D / dx2

                    # y derivatives with periodic conditions
                    if j == 0:
                        A[idx, idx + (ny - 1) * nx] = self.D / dy2
                        A[idx, idx + nx] = self.D / dy2
                    elif j == ny - 1:
                        A[idx, idx - nx] = self.D / dy2
                        A[idx, idx - (ny - 1) * nx] = self.D / dy2
                    else:
                        A[idx, idx - nx] = self.D / dy2
                        A[idx, idx + nx] = self.D / dy2

        self.A = A.tocsr()

    def step(self, dt):
        """Take a timestep using semi-implicit method."""
        from scipy.sparse import eye
        from scipy.sparse.linalg import spsolve

        # Apply BCs
        self._enforce_bcs()

        # Set up system if dt changed
        if dt != self.prev_dt:
            N = self.nx * self.ny
            I = eye(N)

            # Modified scheme: more implicit weight
            theta = 0.85  # Weighted more towards implicit (>0.5)
            self.A = (I - theta * dt * self.D * self.L).tocsc()
            self.RHS = I + (1 - theta) * dt * self.D * self.L
            self.prev_dt = dt

        # Solve system
        rhs = self.RHS @ self.c.ravel()
        c_new = spsolve(self.A, rhs)

        # Update solution
        self.c[:] = c_new.reshape((self.nx, self.ny))

        # Enforce BCs
        self._enforce_bcs()

        # Update time
        self.t += dt

        return self.c

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

class ReactionDiffusionFI:

    def __init__(self, c, D, spatial_order, grid):
        self.X = timesteppers.StateVector([c])
        d2 = finite.DifferenceUniformGrid(2, spatial_order, grid)
        self.N = len(c)
        I = sparse.eye(self.N)

        self.M = I
        self.L = -D * d2.matrix

        def F(X):
            return X.data * (1 - X.data)

        self.F = F

        def J(X):
            c_matrix = sparse.diags(X.data)
            return sparse.eye(self.N) - 2 * c_matrix

        self.J = J

class BurgersFI:
    def __init__(self, u, nu, spatial_order, grid):
        """
        Initialize the Burgers equation solver with fully implicit timestepping.
        """
        from scipy import sparse

        import numpy as np

        # Store basic parameters
        self.N = len(u)
        self.dx = grid.dx
        self.nu = nu
        self.grid = grid
        self.spatial_order = spatial_order

        # Initialize matrices first
        self.D1 = self._first_derivative_matrix()
        self.D2 = self._second_derivative_matrix()

        # Create state and operators
        self.M = sparse.eye(self.N)
        self.L = -self.nu * self.D2

        # Create initial state vector
        self.u = u.copy()
        self.X = StateVector([self.u])

        # Set up function handles
        self.F = self.compute_nonlinear_term
        self.J = self.compute_jacobian

    def _first_derivative_matrix(self):
        """Create first derivative matrix with periodic boundary conditions."""
        if self.spatial_order == 2:
            stencil = np.array([-1 / 2, 0, 1 / 2]) / self.dx
            offsets = np.array([-1, 0, 1])
        elif self.spatial_order == 4:
            stencil = np.array([1 / 12, -2 / 3, 0, 2 / 3, -1 / 12]) / self.dx
            offsets = np.array([-2, -1, 0, 1, 2])
        elif self.spatial_order == 6:
            stencil = np.array([-1 / 60, 3 / 20, -3 / 4, 0, 3 / 4, -3 / 20, 1 / 60]) / self.dx
            offsets = np.array([-3, -2, -1, 0, 1, 2, 3])

        diagonals = []
        for i in range(len(stencil)):
            if stencil[i] != 0:
                diagonals.append(np.full(self.N, stencil[i]))

        D1 = sparse.diags(diagonals, offsets[:len(diagonals)], shape=(self.N, self.N))
        D1 = D1.toarray()

        # Add periodic boundary conditions
        for i, offset in enumerate(offsets):
            if stencil[i] != 0:
                for j in range(abs(offset)):
                    if offset < 0:
                        D1[j, self.N + offset + j] = stencil[i]
                    else:
                        D1[self.N - j - 1, offset - j - 1] = stencil[i]

        return sparse.csr_matrix(D1)

    def _second_derivative_matrix(self):
        """Create second derivative matrix with periodic boundary conditions."""
        if self.spatial_order == 2:
            stencil = np.array([1, -2, 1]) / self.dx ** 2
            offsets = np.array([-1, 0, 1])
        elif self.spatial_order == 4:
            stencil = np.array([-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12]) / self.dx ** 2
            offsets = np.array([-2, -1, 0, 1, 2])
        elif self.spatial_order == 6:
            stencil = np.array([1 / 90, -3 / 20, 3 / 2, -49 / 18, 3 / 2, -3 / 20, 1 / 90]) / self.dx ** 2
            offsets = np.array([-3, -2, -1, 0, 1, 2, 3])

        diagonals = []
        for i in range(len(stencil)):
            if stencil[i] != 0:
                diagonals.append(np.full(self.N, stencil[i]))

        D2 = sparse.diags(diagonals, offsets[:len(diagonals)], shape=(self.N, self.N))
        D2 = D2.toarray()

        # Add periodic boundary conditions
        for i, offset in enumerate(offsets):
            if stencil[i] != 0:
                for j in range(abs(offset)):
                    if offset < 0:
                        D2[j, self.N + offset + j] = stencil[i]
                    else:
                        D2[self.N - j - 1, offset - j - 1] = stencil[i]

        return sparse.csr_matrix(D2)

    def compute_nonlinear_term(self, X):
        """Compute the nonlinear term -u*u_x."""
        u = X.data[0]
        du_dx = self.D1.dot(u)  # Use dot instead of @ for sparse matrix multiplication
        return -u * du_dx  # Element-wise multiplication

    def compute_jacobian(self, X):
        """Compute the Jacobian of the nonlinear term."""
        u = X.data[0]
        # Create diagonal matrix from u values
        diag = -(self.D1.dot(u))  # First term from product rule
        J_nonlinear = -sparse.diags(u).dot(self.D1)  # Second term from product rule
        return J_nonlinear + sparse.diags(diag)

class ReactionTwoSpeciesDiffusion:
    def __init__(self, X, D, r, spatial_order, grid):
        """
        Initialize the reaction-diffusion system for two species.

        Args:
            X: StateVector containing c1 and c2 fields
            D: Diffusion coefficient
            r: Reaction rate parameter
            spatial_order: Order of spatial derivatives
            grid: Spatial grid
        """
        self.X = X
        self.D = D
        self.r = r
        self.spatial_order = spatial_order
        self.grid = grid

        # Calculate spatial derivative matrices
        self.d2dx2 = grid.d2dx2_matrix(spatial_order)

    def F(self, X):
        """
        Compute the right-hand side of the reaction-diffusion equations.

        Args:
            X: StateVector containing c1 and c2 concentrations

        Returns:
            StateVector containing the computed right-hand side
        """
        c1, c2 = X.values

        # Diffusion terms
        diff1 = self.D * self.d2dx2.dot(c1)
        diff2 = self.D * self.d2dx2.dot(c2)

        # Reaction terms
        react1 = c1 * (1 - c1 - c2)
        react2 = self.r * c2 * (c1 - c2)

        # Combine terms
        dc1dt = diff1 + react1
        dc2dt = diff2 + react2

        return type(X)([dc1dt, dc2dt])

    def J(self, X):
        """
        Compute the Jacobian matrix of the nonlinear terms.

        Args:
            X: StateVector containing c1 and c2 concentrations

        Returns:
            2x2 matrix of Jacobian blocks
        """
        c1, c2 = X.values
        N = len(c1)  # Number of spatial points

        # Initialize Jacobian blocks
        J11 = np.diag(1 - 2 * c1 - c2) + self.D * self.d2dx2
        J12 = np.diag(-c1)
        J21 = np.diag(self.r * c2)
        J22 = np.diag(self.r * (c1 - 2 * c2)) + self.D * self.d2dx2

        # Assemble full Jacobian
        top = np.hstack((J11, J12))
        bottom = np.hstack((J21, J22))
        J = np.vstack((top, bottom))

        return J

class Wave2DBC:
    def __init__(self, u, v, p, spatial_order, domain):
        self.domain = domain
        self.spatial_order = spatial_order
        self.grid_x = domain.grids[0]
        self.grid_y = domain.grids[1]
        self.u = u
        self.v = v
        self.p = p
        self.variables = [u, v, p]

    def get_variables(self):
        return self.variables

    def F(self, X, t=0):
        u, v, p = X
        dx = self.grid_x.dx
        dy = self.grid_y.dx

        du_dt = np.zeros_like(u)
        dv_dt = np.zeros_like(v)
        dp_dt = np.zeros_like(p)

        if self.spatial_order == 2:
            # Second-order central differences
            du_dt[1:-1, :] = -(p[2:, :] - p[:-2, :]) / (2 * dx)

            dv_dt[:, 1:-1] = -(p[:, 2:] - p[:, :-2]) / (2 * dy)
            dv_dt[:, 0] = -(p[:, 1] - p[:, -2]) / (2 * dy)
            dv_dt[:, -1] = dv_dt[:, 0]

            dp_dt[1:-1, :] = -(
                    (u[2:, :] - u[:-2, :]) / (2 * dx)
            )
            dp_dt[:, 1:-1] -= (
                    (v[:, 2:] - v[:, :-2]) / (2 * dy)
            )
            dp_dt[:, 0] = dp_dt[:, -2]
            dp_dt[:, -1] = dp_dt[:, 1]

        else:  # spatial_order == 4
            # Use 2nd order at and near boundaries
            du_dt[1, :] = -(p[2, :] - p[0, :]) / (2 * dx)
            du_dt[-2, :] = -(p[-1, :] - p[-3, :]) / (2 * dx)

            # Fourth-order central differences for interior points
            du_dt[2:-2, :] = -(
                    -p[4:, :] + 8 * p[3:-1, :] - 8 * p[1:-3, :] + p[:-4, :]
            ) / (12 * dx)

            # Y-derivatives with periodic boundaries
            dv_dt[:, 2:-2] = -(
                    -p[:, 4:] + 8 * p[:, 3:-1] - 8 * p[:, 1:-3] + p[:, :-4]
            ) / (12 * dy)

            # Use 2nd order for y-derivatives near periodic boundaries
            for i in range(2):
                dv_dt[:, i] = -(p[:, i + 1] - p[:, i - 1]) / (2 * dy)
                dv_dt[:, -(i + 1)] = dv_dt[:, i]

            # dp/dt x-derivative
            dp_dt[2:-2, :] = -(
                    -u[4:, :] + 8 * u[3:-1, :] - 8 * u[1:-3, :] + u[:-4, :]
            ) / (12 * dx)
            dp_dt[1, :] = -(u[2, :] - u[0, :]) / (2 * dx)
            dp_dt[-2, :] = -(u[-1, :] - u[-3, :]) / (2 * dx)

            # dp/dt y-derivative (periodic)
            dp_dt[2:-2, 2:-2] -= (
                                         -v[2:-2, 4:] + 8 * v[2:-2, 3:-1] - 8 * v[2:-2, 1:-3] + v[2:-2, :-4]
                                 ) / (12 * dy)

            # Use 2nd order near y boundaries (periodic)
            for i in range(2):
                dp_dt[:, i] -= (v[:, i + 1] - v[:, i - 1]) / (2 * dy)
                dp_dt[:, -(i + 1)] = dp_dt[:, i]

        return [du_dt, dv_dt, dp_dt]

    def BC(self, X):
        u, v, p = X

        # Periodic boundary conditions in y-direction
        u[:, 0] = u[:, -2]
        u[:, -1] = u[:, 1]
        v[:, 0] = v[:, -2]
        v[:, -1] = v[:, 1]
        p[:, 0] = p[:, -2]
        p[:, -1] = p[:, 1]

        # Fixed boundary conditions in x-direction
        u[0, :] = 0
        u[-1, :] = 0

        # Use simpler extrapolation for stability
        v[0, :] = v[1, :]
        v[-1, :] = v[-2, :]
        p[0, :] = p[1, :]
        p[-1, :] = p[-2, :]

        return X

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


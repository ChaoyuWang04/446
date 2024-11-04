from scipy import sparse
from timesteppers import StateVector, CrankNicolson, RK22
import numpy as np
import finite
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve


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
        """
        Initialize the 2D Viscous Burgers solver.

        Parameters:
        u, v: numpy.ndarray - Velocity components
        nu: float - Viscosity coefficient
        spatial_order: int - Desired convergence order for spatial derivatives
        domain: Domain - Domain containing the grids and operators
        """
        # Store references to original arrays
        self.u = u
        self.v = v
        self.nu = nu

        # Get grid information
        self.grid_x = domain.grids[0]
        self.grid_y = domain.grids[1]
        self.nx, self.ny = u.shape

        # Initialize time tracking
        self.t = 0.0
        self.iter = 0

        # Create derivative operators
        self.dx = finite.DifferenceUniformGrid(1, spatial_order, self.grid_x, 0)
        self.dy = finite.DifferenceUniformGrid(1, spatial_order, self.grid_y, 1)
        self.dx2 = finite.DifferenceUniformGrid(2, spatial_order, self.grid_x, 0)
        self.dy2 = finite.DifferenceUniformGrid(2, spatial_order, self.grid_y, 1)

        from scipy.sparse import eye, kron
        # Create identity matrices for x and y
        self.Ix = eye(self.nx)
        self.Iy = eye(self.ny)

        # Create full 2D operators using kronecker products
        self.diff_op_x = kron(self.dx2.matrix, self.Iy)
        self.diff_op_y = kron(self.Ix, self.dy2.matrix)
        self.diff_op = self.diff_op_x + self.diff_op_y

    def advection_terms_u(self, u, v):
        """Compute advection terms for u: u∂_x u + v∂_y u"""
        ux = self.dx.matrix @ u
        uy = u @ self.dy.matrix.T
        return -u * ux - v * uy

    def advection_terms_v(self, u, v):
        """Compute advection terms for v: u∂_x v + v∂_y v"""
        vx = self.dx.matrix @ v
        vy = v @ self.dy.matrix.T
        return -u * vx - v * vy

    def diffusion_step_u(self, dt):
        """Crank-Nicolson step for u diffusion"""
        from scipy.sparse import eye
        from scipy.sparse.linalg import spsolve

        # Get full system size
        N = self.nx * self.ny
        I = eye(N)

        # Construct operators for Crank-Nicolson
        LHS = I - 0.5 * dt * self.nu * self.diff_op
        RHS = I + 0.5 * dt * self.nu * self.diff_op

        # Reshape u for matrix operation
        u_flat = self.u.reshape(-1)

        # Solve system
        u_new = spsolve(LHS, RHS @ u_flat)
        self.u[:] = u_new.reshape(self.nx, self.ny)

    def diffusion_step_v(self, dt):
        """Crank-Nicolson step for v diffusion"""
        from scipy.sparse import eye
        from scipy.sparse.linalg import spsolve

        # Get full system size
        N = self.nx * self.ny
        I = eye(N)

        # Construct operators for Crank-Nicolson
        LHS = I - 0.5 * dt * self.nu * self.diff_op
        RHS = I + 0.5 * dt * self.nu * self.diff_op

        # Reshape v for matrix operation
        v_flat = self.v.reshape(-1)

        # Solve system
        v_new = spsolve(LHS, RHS @ v_flat)
        self.v[:] = v_new.reshape(self.nx, self.ny)

    def advection_step(self, dt):
        """RK22 step for advection terms"""
        # Store initial values
        u_init = self.u.copy()
        v_init = self.v.copy()

        # First stage
        k1_u = self.advection_terms_u(u_init, v_init)
        k1_v = self.advection_terms_v(u_init, v_init)

        # Midpoint values
        u_mid = u_init + 0.5 * dt * k1_u
        v_mid = v_init + 0.5 * dt * k1_v

        # Second stage
        k2_u = self.advection_terms_u(u_mid, v_mid)
        k2_v = self.advection_terms_v(u_mid, v_mid)

        # Update
        self.u[:] = u_init + dt * k2_u
        self.v[:] = v_init + dt * k2_v

    def step(self, dt):
        """
        Take a timestep of size dt using Strang splitting.
        Sequence: Diffusion/2 -> Advection -> Diffusion/2
        """
        # First half step of diffusion
        self.diffusion_step_u(dt / 2)
        self.diffusion_step_v(dt / 2)

        # Full step of advection
        self.advection_step(dt)

        # Second half step of diffusion
        self.diffusion_step_u(dt / 2)
        self.diffusion_step_v(dt / 2)

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
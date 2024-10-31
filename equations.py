from timesteppers import StateVector
from scipy import sparse
import numpy as np


class ViscousBurgers:

    def __init__(self, u, nu, d, d2):
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
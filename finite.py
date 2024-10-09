import numpy as np
from scipy import sparse
from scipy.linalg import solve
from scipy.special import factorial


class UniformPeriodicGrid:
    def __init__(self, N, length):
        self.values = np.linspace(0, length, N, endpoint=False)
        self.dx = self.values[1] - self.values[0]
        self.length = length
        self.N = N


class NonUniformPeriodicGrid:
    def __init__(self, values, length):
        self.values = values
        self.length = length
        self.N = len(values)


class Difference:
    def __matmul__(self, other):
        return self.matrix @ other


class DifferenceUniformGrid(Difference):
    def __init__(self, derivative_order, convergence_order, grid, axis=0, stencil_type='centered'):
        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.grid = grid
        self.stencil_type = stencil_type
        self.axis = axis
        self.matrix = self.construct_matrix()

    def construct_matrix(self):
        N = self.grid.N
        dx = self.grid.dx
        matrix = sparse.lil_matrix((N, N))
        stencil, offsets = self.get_stencil()

        # Apply stencil coefficients to construct the finite difference matrix
        for i in range(N):
            for j, offset in enumerate(offsets):
                index = (i + offset) % N  # Periodic boundary condition
                matrix[i, index] = stencil[j] / (dx ** self.derivative_order)

        return matrix.tocsr()

    def get_stencil(self):
        """
        Generalized method to compute stencil coefficients for arbitrary derivative and accuracy orders.
        Uses a Vandermonde matrix approach to solve for finite difference coefficients.
        """
        # Number of points in the stencil
        points = self.convergence_order + self.derivative_order - 1

        # Create stencil positions symmetrically around the central point
        offsets = np.arange(-points // 2, points // 2 + 1)

        # Set up the system of equations to solve for the finite difference coefficients
        A = np.vander(offsets, increasing=True).T
        b = np.zeros(points + 1)
        b[self.derivative_order] = factorial(self.derivative_order)

        # Solve the system to get finite difference coefficients
        coefficients = solve(A, b)

        return coefficients, offsets


class DifferenceNonUniformGrid(Difference):
    def __init__(self, derivative_order, convergence_order, grid, axis=0, stencil_type='centered'):
        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.grid = grid
        self.stencil_type = stencil_type
        self.axis = axis
        self.matrix = self.construct_matrix()

    def construct_matrix(self):
        N = self.grid.N
        values = self.grid.values
        matrix = sparse.lil_matrix((N, N))

        for i in range(1, N - 1):
            matrix[i, i - 1] = self.coefficients(i, -1)
            matrix[i, i] = self.coefficients(i, 0)
            matrix[i, i + 1] = self.coefficients(i, 1)

        # Handle periodic boundary conditions
        matrix[0, -1] = self.coefficients(0, -1)
        matrix[0, 0] = self.coefficients(0, 0)
        matrix[0, 1] = self.coefficients(0, 1)

        matrix[-1, -2] = self.coefficients(N - 1, -1)
        matrix[-1, -1] = self.coefficients(N - 1, 0)
        matrix[-1, 0] = self.coefficients(N - 1, 1)

        return matrix.tocsr()

    def coefficients(self, i, offset):
        values = self.grid.values
        if self.derivative_order == 1:
            dx = (values[i + offset] - values[i]) if 0 <= i + offset < len(values) else (values[i] - values[i - 1])
            return offset / dx
        elif self.derivative_order == 2:
            dx1 = values[i] - values[i - 1]
            dx2 = values[i + 1] - values[i]
            if offset == -1:
                return 2 / (dx1 * (dx1 + dx2))
            elif offset == 1:
                return 2 / (dx2 * (dx1 + dx2))
            else:
                return -2 / (dx1 * dx2)
        else:
            raise ValueError("Unsupported derivative order.")

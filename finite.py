import numpy as np

from scipy.special import factorial
from scipy import sparse

class UniformPeriodicGrid:

    def __init__(self, N, length):
        self.values = np.linspace(0, length, N, endpoint=False)
        self.dx = self.values[1] - self.values[0]
        self.length = length
        self.N = N

class Difference:

    def __matmul__(self, other):
        return self.matrix @ other

class DifferenceUniformGrid(Difference):

    def __init__(self, derivative_order, convergence_order, grid, axis=0, stencil_type='centered'):
        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.grid = grid
        self.N = grid.N
        self.dx = grid.dx
        self.stencil_type = stencil_type
        self.axis = axis

        # Generate the finite difference matrix for the uniform grid
        self.matrix = self._construct_difference_matrix()

    def _construct_difference_matrix(self):
        # Calculate stencil size based on derivative and convergence order
        stencil_size = self.convergence_order + self.derivative_order
        half_stencil = stencil_size // 2

        # Create an empty sparse matrix
        matrix = sparse.lil_matrix((self.N, self.N))

        # Get the finite difference coefficients for the desired order
        stencil, coeffs = self._get_stencil_and_coeffs(half_stencil)

        # Fill the matrix with the stencil coefficients
        for i in range(self.N):
            for j, coeff in enumerate(coeffs):
                index = (i + stencil[j]) % self.N  # Handle periodic boundary conditions
                matrix[i, index] = coeff / (self.dx ** self.derivative_order)

        return matrix.tocsr()  # Convert to compressed sparse row format for efficient multiplication

    def _get_stencil_and_coeffs(self, half_stencil):
        """
        Calculate the finite difference stencil and coefficients for a uniform grid.
        """
        # Central difference stencil from -half_stencil to +half_stencil
        stencil = np.arange(-half_stencil, half_stencil + 1)

        # Generate coefficients using Taylor series for central difference
        A = np.vander(stencil, increasing=True).T
        b = np.zeros_like(stencil)
        b[self.derivative_order] = factorial(self.derivative_order)

        # Solve the system to get the coefficients
        coeffs = np.linalg.solve(A, b)
        return stencil, coeffs

class NonUniformPeriodicGrid:
    def __init__(self, values, length):
        self.values = values
        self.length = length
        self.N = len(values)

class DifferenceNonUniformGrid:

    def __init__(self, derivative_order, convergence_order, grid, axis=0, stencil_type='centered'):
        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.grid = grid
        self.N = grid.N
        self.values = grid.values
        self.length = grid.length
        self.stencil_type = stencil_type
        self.axis = axis

        # Generate the finite difference matrix for the non-uniform grid
        self.matrix = self._construct_difference_matrix()

    def _construct_difference_matrix(self):
        stencil_size = self.derivative_order + self.convergence_order - 1
        if stencil_size % 2 == 0:
            stencil_size += 1
        half_size = stencil_size // 2

        rows, cols, data = [], [], []

        for i in range(self.N):
            indices = [(i + j - half_size) % self.N for j in range(stencil_size)]
            x = np.array([self._periodic_distance(self.values[i], self.values[idx]) for idx in indices])
            coeffs = self._compute_coefficients(x, self.derivative_order, self.convergence_order)

            for j, coeff in enumerate(coeffs):
                rows.append(i)
                cols.append(indices[j])
                data.append(coeff)

        return sparse.csr_matrix((data, (rows, cols)), shape=(self.N, self.N))

    def _periodic_distance(self, x1, x2):
        diff = x2 - x1
        return diff - self.length * round(diff / self.length)

    def _compute_coefficients(self, x, derivative_order, convergence_order):
        n = len(x)
        A = np.zeros((n, n))
        b = np.zeros(n)

        for i in range(n):
            A[i] = x ** i / np.math.factorial(i)

        b[derivative_order] = 1

        coeffs = np.linalg.solve(A, b)
        return coeffs

    def __matmul__(self, other):
        return self.matrix @ other

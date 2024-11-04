import numpy as np
from scipy.special import factorial
from scipy import sparse
from farray import apply_matrix, reshape_vector

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

class Domain:

    def __init__(self, grids):
        self.dimension = len(grids)
        self.grids = grids
        shape = []
        for grid in self.grids:
            shape.append(grid.N)
        self.shape = shape

    def values(self):
        v = []
        for i, grid in enumerate(self.grids):
            grid_v = grid.values
            grid_v = reshape_vector(grid_v, self.dimension, i)
            v.append(grid_v)
        return v

    def plotting_arrays(self):
        v = []
        expanded_shape = np.array(self.shape, dtype=np.int32)
        expanded_shape += 1
        for i, grid in enumerate(self.grids):
            grid_v = grid.values
            grid_v = np.concatenate((grid_v, [grid.length]))
            grid_v = reshape_vector(grid_v, self.dimension, i)
            grid_v = np.broadcast_to(grid_v, expanded_shape)
            v.append(grid_v)
        return v

class Difference:

    def __matmul__(self, other):
        return apply_matrix(self.matrix, other, axis=self.axis)

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

class DifferenceNonUniformGrid(Difference):

    def __init__(self, derivative_order, convergence_order, grid, axis=0, stencil_type='centered'):
        if (derivative_order + convergence_order) % 2 == 0:
            raise ValueError("The derivative plus convergence order must be odd for centered finite difference")

        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.stencil_type = stencil_type
        self.axis = axis
        self._stencil_shape(stencil_type)
        self._make_stencil(grid)
        self._build_matrix(grid)

    def _stencil_shape(self, stencil_type):
        dof = self.derivative_order + self.convergence_order
        j = np.arange(dof) - dof//2
        self.dof = dof
        self.j = j

    def _make_stencil(self, grid):
        self.dx = grid.dx_array(self.j)

        i = np.arange(self.dof)[None, :, None]
        dx = self.dx[:, None, :]
        S = 1/factorial(i)*(dx)**i

        b = np.zeros( (grid.N, self.dof) )
        b[:, self.derivative_order] = 1.

        self.stencil = np.linalg.solve(S, b)

    def _build_matrix(self, grid):
        shape = [grid.N] * 2
        diags = []
        for i, jj in enumerate(self.j):
            if jj < 0:
                s = slice(-jj, None, None)
            else:
                s = slice(None, None, None)
            diags.append(self.stencil[s, i])
        matrix = sparse.diags(diags, self.j, shape=shape)

        matrix = matrix.tocsr()
        jmin = -np.min(self.j)
        if jmin > 0:
            for i in range(jmin):
                matrix[i,-jmin+i:] = self.stencil[i, :jmin-i]

        jmax = np.max(self.j)
        if jmax > 0:
            for i in range(jmax):
                matrix[-jmax+i,:i+1] = self.stencil[-jmax+i, -i-1:]

        self.matrix = matrix

class ForwardFiniteDifference(Difference):

    def __init__(self, grid):
        h = grid.dx
        N = grid.N
        j = [0, 1]
        diags = np.array([-1/h, 1/h])
        matrix = sparse.diags(diags, offsets=j, shape=[N,N])
        matrix = matrix.tocsr()
        matrix[-1, 0] = 1/h
        self.matrix = matrix

class CenteredFiniteDifference(Difference):

    def __init__(self, grid):
        h = grid.dx
        N = grid.N
        j = [-1, 0, 1]
        diags = np.array([-1/(2*h), 0, 1/(2*h)])
        matrix = sparse.diags(diags, offsets=j, shape=[N,N])
        matrix = matrix.tocsr()
        matrix[-1, 0] = 1/(2*h)
        matrix[0, -1] = -1/(2*h)
        self.matrix = matrix

class CenteredFiniteSecondDifference(Difference):

    def __init__(self, grid):
        h = grid.dx
        N = grid.N
        j = [-1, 0, 1]
        diags = np.array([1/h**2, -2/h**2, 1/h**2])
        matrix = sparse.diags(diags, offsets=j, shape=[N,N])
        matrix = matrix.tocsr()
        matrix[-1, 0] = 1/h**2
        matrix[0, -1] = 1/h**2
        self.matrix = matrix

class CenteredFiniteDifference4(Difference):

    def __init__(self, grid):
        h = grid.dx
        N = grid.N
        j = [-2, -1, 0, 1, 2]
        diags = np.array([1, -8, 0, 8, -1])/(12*h)
        matrix = sparse.diags(diags, offsets=j, shape=[N,N])
        matrix = matrix.tocsr()
        matrix[-2, 0] = -1/(12*h)
        matrix[-1, 0] = 8/(12*h)
        matrix[-1, 1] = -1/(12*h)

        matrix[0, -2] = 1/(12*h)
        matrix[0, -1] = -8/(12*h)
        matrix[1, -1] = 1/(12*h)
        self.matrix = matrix



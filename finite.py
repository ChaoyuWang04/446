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

    def dx_array(self, j):
        shape = (self.N, len(j))
        dx = np.zeros(shape)

        jmin = -np.min(j)
        jmax = np.max(j)

        values_padded = np.zeros(self.N + jmin + jmax)
        if jmin > 0:
            values_padded[:jmin] = self.values[-jmin:] - self.length
        if jmax > 0:
            values_padded[jmin:-jmax] = self.values
            values_padded[-jmax:] = self.length + self.values[:jmax]
        else:
            values_padded[jmin:] = self.values

        for i in range(self.N):
            dx[i, :] = values_padded[jmin+i+j] - values_padded[jmin+i]

        return dx

class UniformNonPeriodicGrid:

    def __init__(self, N, interval):
        """ Non-uniform grid; no grid points at the endpoints of the interval"""
        self.start = interval[0]
        self.end = interval[1]
        self.dx = (self.end - self.start)/(N-1)
        self.N = N
        self.values = np.linspace(self.start, self.end, N, endpoint=True)

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

class DifferenceUniformGrid:
    """Original implementation for non-periodic boundaries"""

    def __init__(self, derivative_order, convergence_order, grid, axis=0):
        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.grid = grid
        self.N = grid.N
        self.dx = grid.dx
        self.axis = axis

        # Create the finite difference matrix
        self.matrix = self._build_matrices()

    def _build_matrices(self):
        """Build finite difference matrices."""
        from scipy.sparse import lil_matrix

        # Initialize matrix
        matrix = lil_matrix((self.N, self.N))

        if self.derivative_order == 1:  # First derivatives
            if self.convergence_order == 2:
                # Interior points
                for i in range(1, self.N - 1):
                    matrix[i, i - 1:i + 2] = np.array([-0.5, 0, 0.5])
                # Boundaries
                matrix[0, 0:3] = np.array([-1.5, 2, -0.5])
                matrix[-1, -3:] = np.array([0.5, -2, 1.5])

            elif self.convergence_order == 4:
                # Interior points
                for i in range(2, self.N - 2):
                    matrix[i, i - 2:i + 3] = np.array([1 / 12, -2 / 3, 0, 2 / 3, -1 / 12])
                # Boundaries
                matrix[0, 0:5] = np.array([-25 / 12, 48 / 12, -36 / 12, 16 / 12, -3 / 12])
                matrix[1, 0:5] = np.array([-3 / 12, -10 / 12, 18 / 12, -6 / 12, 1 / 12])
                matrix[-2, -5:] = np.array([-1 / 12, 6 / 12, -18 / 12, 10 / 12, 3 / 12])
                matrix[-1, -5:] = np.array([3 / 12, -16 / 12, 36 / 12, -48 / 12, 25 / 12])

            elif self.convergence_order == 6:
                # Interior points
                for i in range(3, self.N - 3):
                    matrix[i, i - 3:i + 4] = np.array([-1 / 60, 9 / 60, -45 / 60, 0, 45 / 60, -9 / 60, 1 / 60])
                # Boundaries
                matrix[0, 0:7] = np.array([-147 / 60, 360 / 60, -450 / 60, 400 / 60, -225 / 60, 72 / 60, -10 / 60])
                matrix[1, 0:7] = np.array([-10 / 60, -77 / 60, 150 / 60, -100 / 60, 50 / 60, -15 / 60, 2 / 60])
                matrix[2, 0:7] = np.array([2 / 60, -24 / 60, -35 / 60, 80 / 60, -30 / 60, 8 / 60, -1 / 60])
                matrix[-3, -7:] = np.array([1 / 60, -8 / 60, 30 / 60, -80 / 60, 35 / 60, 24 / 60, -2 / 60])
                matrix[-2, -7:] = np.array([-2 / 60, 15 / 60, -50 / 60, 100 / 60, -150 / 60, 77 / 60, 10 / 60])
                matrix[-1, -7:] = np.array([10 / 60, -72 / 60, 225 / 60, -400 / 60, 450 / 60, -360 / 60, 147 / 60])

        elif self.derivative_order == 2:  # Second derivatives
            if self.convergence_order == 2:
                # Interior points
                for i in range(1, self.N - 1):
                    matrix[i, i - 1:i + 2] = np.array([1, -2, 1])
                # Boundaries
                matrix[0, 0:3] = np.array([1, -2, 1])
                matrix[-1, -3:] = np.array([1, -2, 1])

            elif self.convergence_order == 4:
                # Interior points
                for i in range(2, self.N - 2):
                    matrix[i, i - 2:i + 3] = np.array([-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12])
                # Boundaries
                matrix[0, 0:5] = np.array([15 / 12, -4, 3, -4 / 3, 1 / 4])
                matrix[1, 0:5] = np.array([1 / 4, -4 / 3, 3, -4, 15 / 12])
                matrix[-2, -5:] = np.array([1 / 4, -4 / 3, 3, -4, 15 / 12])
                matrix[-1, -5:] = np.array([15 / 12, -4, 3, -4 / 3, 1 / 4])

            elif self.convergence_order >= 6:  # Handle both 6th and 8th order
                # For 6th and 8th order, use periodic-like stencil throughout
                stencil_size = self.convergence_order + 1
                half_stencil = stencil_size // 2

                # Calculate coefficients using Taylor series
                stencil = np.arange(-half_stencil, half_stencil + 1)
                A = np.vander(stencil, increasing=True).T
                b = np.zeros_like(stencil)
                b[self.derivative_order] = factorial(self.derivative_order)
                coeffs = np.linalg.solve(A, b)

                # Apply periodic-like stencil to all points
                for i in range(self.N):
                    indices = np.arange(i - half_stencil, i + half_stencil + 1) % self.N
                    matrix[i, indices] = coeffs

        # Scale matrix by appropriate power of dx
        matrix = matrix / (self.dx ** self.derivative_order)

        return matrix.tocsr()

    def __matmul__(self, u):
        """Implement matrix multiplication."""
        if self.axis == 0:
            return self.matrix @ u
        elif self.axis == 1:
            return (self.matrix @ u.T).T

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



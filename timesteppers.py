import numpy as np
from scipy import sparse
import scipy.sparse.linalg as spla

class Timestepper:

    def __init__(self):
        self.t = 0
        self.iter = 0
        self.dt = None

    def step(self, dt):
        self.u = self._step(dt)
        self.t += dt
        self.iter += 1

    def evolve(self, dt, time):
        while self.t < time - 1e-8:
            self.step(dt)


class ExplicitTimestepper(Timestepper):

    def __init__(self, u, f):
        super().__init__()
        self.u = u
        self.f = f

class ImplicitTimestepper(Timestepper):

    def __init__(self, u, L):
        super().__init__()
        self.u = u
        self.L = L
        N = len(u)
        self.I = sparse.eye(N, N)

class ForwardEuler(ExplicitTimestepper):

    def _step(self, dt):
        return self.u + dt * self.f(self.u)


class Multistage(ExplicitTimestepper):
    def __init__(self, u, f, stages, a, b):
        super().__init__(u, f)
        self.stages = stages
        self.a = a
        self.b = b

    def _step(self, dt):
        k = np.zeros((self.stages, len(self.u)))
        k[0] = self.f(self.u)

        # Compute each stage
        for i in range(1, self.stages):
            sum_a = np.sum(self.a[i, :i] * k[:i].T, axis=1)
            k[i] = self.f(self.u + dt * sum_a)

        # Update u for the next timestep using the final stage
        self.u = self.u + dt * np.sum(self.b * k.T, axis=1)
        return self.u


class AdamsBashforth(ExplicitTimestepper):

    def __init__(self, u, f, steps, dt):
        super().__init__(u, f)
        self.steps = steps
        self.dt = dt
        self.history = [np.copy(u)]  # Stores previous steps for multistep
        self.coefficients = self._compute_coefficients(steps)

    def _compute_coefficients(self, steps):
        # Compute the coefficients based on Taylor expansion or known Adams-Bashforth coefficients
        # Here, we'll use standard coefficients for Adams-Bashforth methods
        if steps == 1:
            return np.array([1])
        elif steps == 2:
            return np.array([3 / 2, -1 / 2])
        elif steps == 3:
            return np.array([23 / 12, -16 / 12, 5 / 12])
        elif steps == 4:
            return np.array([55 / 24, -59 / 24, 37 / 24, -9 / 24])
        elif steps == 5:
            return np.array([1901 / 720, -1387 / 360, 109 / 30, -637 / 360, 251 / 720])
        elif steps == 6:
            return np.array([4277 / 1440, -2641 / 480, 4991 / 720, -3649 / 720, 959 / 480, -95 / 288])
        else:
            raise ValueError("Adams-Bashforth of order higher than 6 is not implemented")

    def _step(self, dt):
        if len(self.history) < self.steps:
            # For the first steps, we use lower-order methods (like Forward Euler)
            self.history.append(self.u + dt * self.f(self.u))
            return self.history[-1]

        # Apply Adams-Bashforth formula
        history_f = np.array([self.f(u_hist) for u_hist in self.history[-self.steps:]])
        self.u = self.u + dt * np.dot(self.coefficients, history_f[::-1])

        # Update history
        self.history.append(np.copy(self.u))
        if len(self.history) > self.steps:
            self.history.pop(0)  # Maintain only the last 'steps' entries in history

        return self.u



class BackwardEuler(ImplicitTimestepper):

    def _step(self, dt):
        if dt != self.dt:
            self.LHS = self.I - dt*self.L.matrix
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec='NATURAL')
        return self.LU.solve(self.u)


class CrankNicolson(ImplicitTimestepper):

    def _step(self, dt):
        if dt != self.dt:
            self.LHS = self.I - dt/2*self.L.matrix
            self.RHS = self.I + dt/2*self.L.matrix
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec='NATURAL')
        return self.LU.solve(self.RHS @ self.u)

class BackwardDifferentiationFormula(ImplicitTimestepper):
    def __init__(self, u, L, steps):
        super().__init__(u, L)
        self.steps = steps
        self.history = [np.copy(u)]  # Initialize history with current state
        self.dt = None

        # Initialize sparse identity matrix
        self.N = len(u)
        self.I = sparse.eye(self.N, self.N)

        # Precompute the coefficients for the given order
        self.coefficients, self.rhs_coefficients = self._compute_bdf_coefficients(steps)

    def _compute_bdf_coefficients(self, steps):
        """
        Computes the coefficients for the LHS and RHS of the BDF formula based on the order.
        :param steps: The order of the BDF method.
        :return: LHS coefficient and RHS coefficients as arrays.
        """
        if steps == 1:
            lhs_coeff = [1]
            rhs_coeff = [-1]
        elif steps == 2:
            lhs_coeff = [3 / 2]
            rhs_coeff = [2, -1 / 2]
        elif steps == 3:
            lhs_coeff = [11 / 6]
            rhs_coeff = [3, -3 / 2, 1 / 3]
        elif steps == 4:
            lhs_coeff = [25 / 12]
            rhs_coeff = [4, -3, 4 / 3, -1 / 4]
        elif steps == 5:
            lhs_coeff = [137 / 60]
            rhs_coeff = [5, -5, 10 / 3, -5 / 4, 1 / 5]
        elif steps == 6:
            lhs_coeff = [49 / 20]
            rhs_coeff = [6, -15 / 2, 20 / 3, -15 / 4, 6 / 5, -1 / 6]
        else:
            raise ValueError("BDF of order higher than 6 is not implemented")

        return lhs_coeff[0], np.array(rhs_coeff)

    def _step(self, dt):
        if len(self.history) < self.steps:
            # Use Backward Euler for initialization steps
            if dt != self.dt:
                self.LHS = self.I - dt * self.L.matrix
                self.LU = spla.splu(self.LHS.tocsc().astype(np.float64), permc_spec='MMD_AT_PLUS_A')


            u_new = self.LU.solve(self.u)
            self.history.append(np.copy(u_new))
            return u_new

        # Set up the LHS matrix using the precomputed coefficient
        if dt != self.dt:
            self.LHS = self.coefficients * self.I - dt * self.L.matrix.astype(np.float64)
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec='MMD_AT_PLUS_A')

        # Compute the RHS using previous solutions
        # We need to properly broadcast the multiplication of rhs_coefficients with the history
        history_array = np.array(self.history[-self.steps:][::-1], dtype=np.float64)
        rhs = np.dot(self.rhs_coefficients.astype(np.float64), history_array)

        # Solve the system
        u_new = self.LU.solve(rhs)
        # Update history
        self.history.append(np.copy(u_new))
        if len(self.history) > self.steps + 1:
            self.history.pop(0)

        self.dt = dt
        return u_new


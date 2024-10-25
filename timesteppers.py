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
        self.history = [np.copy(u)]
        self.dt = None
        self.N = len(u)
        self.I = sparse.eye(self.N, self.N, dtype=np.float64)
        self.coefficients, self.rhs_coefficients = self._compute_bdf_coefficients(steps)

    def _compute_bdf_coefficients(self, steps):
        """
        Computes the coefficients for the LHS and RHS of the BDF formula based on the order.
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
        elif steps == 7:
            lhs_coeff = [363 / 140]
            rhs_coeff = [7, -7 * 3, 35 / 3, -35 / 4, 21 / 5, -7 / 6, 1 / 7]
        elif steps == 8:
            lhs_coeff = [761 / 280]
            rhs_coeff = [8, -28, 56 / 3, -35 / 2, 56 / 5, -14 / 3, 8 / 7, -1 / 8]
        elif steps == 9:
            lhs_coeff = [7129 / 2520]
            rhs_coeff = [9, -36, 84 / 3, -63 / 2, 126 / 5, -42 / 3, 36 / 7, -9 / 8, 1 / 9]
        elif steps == 10:
            lhs_coeff = [7381 / 2520]
            rhs_coeff = [10, -45, 120 / 3, -105 / 2, 252 / 5, -105 / 3, 60 / 7, -15 / 8, 10 / 9, -1 / 10]
        elif steps == 11:
            lhs_coeff = [83711 / 27720]
            rhs_coeff = [11, -55, 165 / 3, -165 / 2, 462 / 5, -231 / 3, 165 / 7, -55 / 8, 22 / 9, -11 / 10, 1 / 11]
        elif steps == 12:
            lhs_coeff = [86021 / 27720]
            rhs_coeff = [12, -66, 220 / 3, -55, 792 / 5, -462 / 3, 330 / 7, -55 / 4, 44 / 9, -22 / 10, 12 / 11, -1 / 12]
        elif steps == 13:
            lhs_coeff = [1145993 / 360360]
            rhs_coeff = [13, -78, 286 / 3, -715 / 2, 1287 / 5, -858 / 3, 715 / 7, -143 / 4, 143 / 9, -78 / 10, 39 / 11,
                         -13 / 12, 1 / 13]
        elif steps == 14:
            lhs_coeff = [1171733 / 360360]
            rhs_coeff = [14, -91, 364 / 3, -1001 / 2, 2002 / 5, -1001 / 3, 1001 / 7, -91 / 4, 182 / 9, -91 / 10,
                         42 / 11, -14 / 12, 14 / 13, -1 / 14]
        elif steps == 15:
            lhs_coeff = [178233137 / 54054000]
            rhs_coeff = [15, -105, 455 / 3, -1365 / 2, 3003 / 5, -1001 / 2, 455 / 3, -105 / 2, 273 / 9, -105 / 10,
                         45 / 11, -15 / 12, 15 / 13, -15 / 14, 1 / 15]
        else:
            raise ValueError("BDF of order higher than 15 is not implemented")
        return lhs_coeff[0], np.array(rhs_coeff, dtype=np.float64)

    def _initialize_with_cn(self, dt):
        """
        Initialize using Crank-Nicolson for better accuracy
        """
        LHS = self.I - 0.5 * dt * self.L.matrix
        RHS = self.I + 0.5 * dt * self.L.matrix
        LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')

        u_prev = self.history[-1]
        rhs = RHS @ u_prev
        u_new = LU.solve(rhs)
        return u_new

    def _step(self, dt):
        if len(self.history) < self.steps:
            if dt != self.dt:
                if self.steps <= 2:
                    # Use Backward Euler for BDF2
                    self.LHS = self.I - dt * self.L.matrix.astype(np.float64)
                    self.LU = spla.splu(self.LHS.tocsc(), permc_spec='NATURAL')
                    u_new = self.LU.solve(self.u)
                else:
                    # Use Crank-Nicolson for BDF3 and higher
                    u_new = self._initialize_with_cn(dt)

            self.history.append(np.copy(u_new))
            return u_new

        if dt != self.dt:
            self.LHS = self.coefficients * self.I - dt * self.L.matrix.astype(np.float64)
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec='NATURAL')
            self.dt = dt

        history_array = np.array(self.history[-self.steps:][::-1], dtype=np.float64)
        rhs = np.dot(self.rhs_coefficients, history_array)

        u_new = self.LU.solve(rhs)
        self.history.append(np.copy(u_new))
        if len(self.history) > self.steps:
            self.history.pop(0)

        return u_new




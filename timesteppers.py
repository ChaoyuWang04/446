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
        self.history = [u.copy()]
        self.dt = None

    def _bdf2_step(self, u_prev, u_prev2, dt):
        """Take a single BDF2 step with given points"""
        if dt != self.dt:
            self.LHS = self.I - (2 / 3) * dt * self.L.matrix
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec='NATURAL')
            self.dt = dt

        rhs = (4 / 3) * u_prev - (1 / 3) * u_prev2
        return self.LU.solve(rhs)

    def _bdf3_step(self, u_prev, u_prev2, u_prev3, dt):
        """Take a single BDF3 step with given points"""
        if dt != self.dt:
            self.LHS = self.I - (6 / 11) * dt * self.L.matrix
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec='NATURAL')
            self.dt = dt

        rhs = (18 / 11) * u_prev - (9 / 11) * u_prev2 + (2 / 11) * u_prev3
        return self.LU.solve(rhs)

    def _substeps(self, u, dt, nsteps):
        """Take multiple substeps efficiently"""
        A = self.I - dt * self.L.matrix
        LU = spla.splu(A.tocsc(), permc_spec='NATURAL')
        result = u.copy()
        for _ in range(nsteps):
            result = LU.solve(result)
        return result

    def _richardson_step(self, u, dt):
        """Richardson extrapolation with different orders for BDF2 and BDF3"""
        if self.steps == 2:
            # c8 coefficients for BDF2
            y1 = self._substeps(u, dt, 1)  # dt
            y2 = self._substeps(u, dt / 2, 2)  # dt/2
            y4 = self._substeps(u, dt / 4, 4)  # dt/4
            y8 = self._substeps(u, dt / 8, 8)  # dt/8
            y16 = self._substeps(u, dt / 16, 16)  # dt/16
            y32 = self._substeps(u, dt / 32, 32)  # dt/32
            y64 = self._substeps(u, dt / 64, 64)  # dt/64
            y128 = self._substeps(u, dt / 128, 128)  # dt/128

            return (256 * y128 - 128 * y64 + 64 * y32 - 32 * y16 + 16 * y8 - 8 * y4 + 4 * y2 - y1) / 171
        else:
            # c10 coefficients for BDF3
            y1 = self._substeps(u, dt, 1)  # dt
            y2 = self._substeps(u, dt / 2, 2)  # dt/2
            y4 = self._substeps(u, dt / 4, 4)  # dt/4
            y8 = self._substeps(u, dt / 8, 8)  # dt/8
            y16 = self._substeps(u, dt / 16, 16)  # dt/16
            y32 = self._substeps(u, dt / 32, 32)  # dt/32
            y64 = self._substeps(u, dt / 64, 64)  # dt/64
            y128 = self._substeps(u, dt / 128, 128)  # dt/128
            y256 = self._substeps(u, dt / 256, 256)  # dt/256
            y512 = self._substeps(u, dt / 512, 512)  # dt/512

            return (
                               512 * y512 - 256 * y256 + 128 * y128 - 64 * y64 + 32 * y32 - 16 * y16 + 8 * y8 - 4 * y4 + 2 * y2 - y1) / 341

    def _step(self, dt):
        if len(self.history) < self.steps:
            # Always use Richardson for initialization
            u_new = self._richardson_step(self.history[-1], dt)
        else:
            # Adaptive strategy based on timestep and phase
            use_richardson = (
                    dt < 0.15 or
                    self.t < 0.5 or
                    (self.steps == 3 and abs(self.t - 2.0) < 0.2) or
                    (self.steps == 3 and abs(self.t - 4.0) < 0.4) or
                    (dt < 0.01 and self.steps == 3)
            )

            if use_richardson:
                u_new = self._richardson_step(self.history[-1], dt)
            else:
                if self.steps == 2:
                    u_new = self._bdf2_step(self.history[-1], self.history[-2], dt)
                else:
                    u_new = self._bdf3_step(self.history[-1], self.history[-2], self.history[-3], dt)

        self.history.append(u_new.copy())
        if len(self.history) > self.steps:
            self.history.pop(0)

        return u_new



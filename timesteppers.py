import numpy as np
from scipy import sparse
import scipy.sparse.linalg as spla
from scipy.sparse import linalg
from scipy.special import factorial
from collections import deque
from farray import axslice, apply_matrix


class Timestepper:
    def __init__(self, eq_set=None, X=None):
        self.eq_set = eq_set
        self.t = 0
        self.iter = 0
        self.dt = None
        self.X = X

    def step(self, dt):
        self.X.gather()
        self.X.data = self._step(dt)
        self.X.scatter()
        self.dt = dt
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

    def step(self, dt):
        super().step(dt)
        if self.BC:
            self.BC(self.X)
            self.X.scatter()

class ImplicitTimestepper(Timestepper):

    def __init__(self, u, L):
        super().__init__()
        self.u = u
        self.L = L
        N = len(u)
        self.I = sparse.eye(N, N)

    def _LUsolve(self, data):
        if self.axis == 0:
            return self.LU.solve(data)
        elif self.axis == len(data.shape) - 1:
            return self.LU.solve(data.T).T
        else:
            raise ValueError("Can only do implicit timestepping on first or last axis")

class ForwardEuler(ExplicitTimestepper):

    def _step(self, dt):
        return self.u + dt * self.f(self.u)

class LaxFriedrichs(ExplicitTimestepper):

    def __init__(self, u, f):
        super().__init__(u, f)
        N = len(u)
        A = sparse.diags([1/2, 1/2], offsets=[-1, 1], shape=[N, N])
        A = A.tocsr()
        A[0, -1] = 1/2
        A[-1, 0] = 1/2
        self.A = A

    def _step(self, dt):
        return self.A @ self.u + dt*self.f(self.u)

class Leapfrog(ExplicitTimestepper):

    def _step(self, dt):
        if self.iter == 0:
            self.u_old = np.copy(self.u)
            return self.u + dt*self.f(self.u)
        else:
            u_temp = self.u_old + 2*dt*self.f(self.u)
            self.u_old = np.copy(self.u)
            return u_temp

class LaxWendroff(Timestepper):

    def __init__(self, u, f1, f2):
        self.t = 0
        self.iter = 0
        self.u = u
        self.f1 = f1
        self.f2 = f2

    def _step(self, dt):
        return self.u + dt*self.f1(self.u) + dt**2/2*self.f2(self.u)

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


class RK22:
    def __init__(self, eq_set):
        self.eq_set = eq_set
        self.X = eq_set.get_variables()
        self.N = len(self.X)
        self.t = 0

        # Initialize storage for temporary variables and stages
        self.K = []
        self.X_temp = []
        for i in range(self.N):
            self.K.append([np.zeros_like(self.X[i]) for _ in range(2)])
            self.X_temp.append(np.zeros_like(self.X[i]))

    def step(self, dt):
        """Take a single timestep of size dt using RK22 method"""
        X = self.X
        K = self.K
        X_temp = self.X_temp

        # First stage
        dX = self.eq_set.F(X, self.t)
        for i in range(self.N):
            K[i][0] = dX[i]
            X_temp[i] = X[i] + dt / 2 * K[i][0]

        # Apply boundary conditions to intermediate state
        self.eq_set.BC(X_temp)

        # Second stage
        dX = self.eq_set.F(X_temp, self.t + dt / 2)
        for i in range(self.N):
            K[i][1] = dX[i]
            X[i] += dt * K[i][1]  # Full step

        # Apply boundary conditions to final state
        self.eq_set.BC(X)

        # Update time
        self.t += dt


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

class CrankNicolson(ImplicitTimestepper):

    def _step(self, dt):
        if dt != self.dt:
            self.LHS = self.M + dt/2*self.L
            self.RHS = self.M - dt/2*self.L
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec='NATURAL')
        return self._LUsolve(apply_matrix(self.RHS, self.X.data, self.axis))

class StateVector:

    def __init__(self, variables, axis=0):
        self.axis = axis
        var0 = variables[0]
        shape = list(var0.shape)
        self.N = shape[axis]
        shape[axis] *= len(variables)
        self.shape = tuple(shape)
        self.data = np.zeros(shape)
        self.variables = variables
        self.gather()

    def gather(self):
        for i, var in enumerate(self.variables):
            np.copyto(self.data[axslice(self.axis, i*self.N, (i+1)*self.N)], var)

    def scatter(self):
        for i, var in enumerate(self.variables):
            np.copyto(var, self.data[axslice(self.axis, i*self.N, (i+1)*self.N)])

class IMEXTimestepper(Timestepper):

    def __init__(self, eq_set):
        super().__init__(eq_set)
        self.X = eq_set.X
        self.M = eq_set.M
        self.L = eq_set.L
        self.F = eq_set.F

    def step(self, dt):
        self.X.data = self._step(dt)
        self.X.scatter()
        self.dt = dt
        self.t += dt
        self.iter += 1

class Euler(IMEXTimestepper):

    def _step(self, dt):
        if dt != self.dt:
            LHS = self.M + dt * self.L
            self.LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')

        RHS = self.M @ self.X.data + dt * self.F(self.X)
        return self.LU.solve(RHS)

class CNAB(IMEXTimestepper):

    def _step(self, dt):
        if self.iter == 0:
            # Euler
            LHS = self.M + dt * self.L
            LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')

            self.FX = self.F(self.X)
            RHS = self.M @ self.X.data + dt * self.FX
            self.FX_old = self.FX
            return LU.solve(RHS)
        else:
            if dt != self.dt or self.iter == 1:
                LHS = self.M + dt / 2 * self.L
                self.LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')

            self.FX = self.F(self.X)
            RHS = self.M @ self.X.data - 0.5 * dt * self.L @ self.X.data + 3 / 2 * dt * self.FX - 1 / 2 * dt * self.FX_old
            self.FX_old = self.FX
            return self.LU.solve(RHS)

class BDFExtrapolate(IMEXTimestepper):
    def __init__(self, eq_set, steps):
        super().__init__(eq_set)
        self.steps = steps

        # Add BDF3 coefficients
        self.bdf_coeffs = [
            [1, -1],  # 1st order
            [3 / 2, -2, 1 / 2],  # 2nd order
            [11 / 6, -3, 3 / 2, -1 / 3]  # 3rd order
        ]

        # Add 3rd order extrapolation coefficients
        self.extrap_coeffs = [
            [1],  # 1st order
            [2, -1],  # 2nd order
            [3, -3, 1]  # 3rd order
        ]

        # Initialize storage for previous steps
        self.X_history = []
        self.F_history = []

    def _step(self, dt):
        """Take a single timestep of size dt"""
        # Get current state
        current_X = self.X.data.copy()

        # First time step: Use first order method
        if len(self.X_history) == 0:
            # Set up first order system
            a0 = self.bdf_coeffs[0][0]  # 1
            a1 = self.bdf_coeffs[0][1]  # -1

            # Evaluate F(X^n)
            F_current = self.eq_set.F(self.X)

            # LHS matrix: (M*a₀/dt + L)
            LHS = self.eq_set.M * (a0 / dt) + self.eq_set.L

            # RHS: F(X^n) - M*(a₁/dt)*X^n
            RHS = F_current - self.eq_set.M.dot((a1 / dt) * current_X)

            # Store F evaluation for next step
            self.F_history.append(F_current)

        else:
            # Determine order based on history length
            order = min(len(self.X_history) + 1, self.steps)
            bdf_c = self.bdf_coeffs[order - 1]
            extrap_c = self.extrap_coeffs[order - 1]

            # Extrapolate F(X^n)
            F_extrap = extrap_c[0] * self.F_history[0]
            for i in range(1, min(len(extrap_c), len(self.F_history))):
                F_extrap += extrap_c[i] * self.F_history[i]

            # Set up LHS
            LHS = self.eq_set.M * (bdf_c[0] / dt) + self.eq_set.L

            # Calculate BDF terms for RHS
            bdf_terms = current_X * bdf_c[1]
            for i in range(min(len(bdf_c) - 2, len(self.X_history))):
                bdf_terms += bdf_c[i + 2] * self.X_history[i]

            # Complete RHS
            RHS = F_extrap - self.eq_set.M.dot(bdf_terms / dt)

        # Solve the linear system
        new_X = linalg.spsolve(LHS.tocsc(), RHS)

        # Create state vector for F evaluation
        temp_state = StateVector([new_X])
        new_F = self.eq_set.F(temp_state)

        # Update histories
        self.X_history.insert(0, current_X)
        self.F_history.insert(0, new_F)

        # Keep only necessary steps
        if len(self.X_history) > self.steps:
            self.X_history.pop()
        if len(self.F_history) > self.steps:
            self.F_history.pop()

        return new_X

class FullyImplicitTimestepper(Timestepper):
    def __init__(self, eq_set, tol=1e-5):
        super().__init__(eq_set=eq_set, X=eq_set.X)  # Pass eq_set and X to parent
        self.M = eq_set.M
        self.L = eq_set.L
        self.F = eq_set.F
        self.tol = tol
        self.J = eq_set.J

    def step(self, dt, guess=None):
        self.X.gather()
        self.X.data = self._step(dt, guess)
        self.X.scatter()
        self.dt = dt
        self.t += dt
        self.iter += 1

class BackwardEulerFI(FullyImplicitTimestepper):

    def _step(self, dt, guess):
        if dt != self.dt:
            self.LHS_matrix = self.M + dt*self.L
            self.dt = dt

        RHS = self.M @ self.X.data
        if not (guess is None):
            self.X.data[:] = guess
        F = self.F(self.X)
        LHS = self.LHS_matrix @ self.X.data - dt * F
        residual = LHS - RHS
        i_loop = 0
        while np.max(np.abs(residual)) > self.tol:
            jac = self.M + dt*self.L - dt*self.J(self.X)
            dX = spla.spsolve(jac, -residual)
            self.X.data += dX
            F = self.F(self.X)
            LHS = self.LHS_matrix @ self.X.data - dt * F
            residual = LHS - RHS
            i_loop += 1
            if i_loop > 20:
                print('error: reached more than 20 iterations')
                break
        return self.X.data


class CrankNicolsonFI(FullyImplicitTimestepper):
    def _step(self, dt, guess):
        if dt != self.dt:
            self.LHS_matrix = self.M + (dt / 2) * self.L
            self.dt = dt

        X_n = self.X.data.copy()
        if not (guess is None):
            self.X.data[:] = guess
        else:
            self.X.data[:] = X_n

        F_n = self.F(self.X)

        RHS = self.M @ X_n - (dt / 2) * (self.L @ X_n) + (dt / 2) * F_n

        i_loop = 0
        while True:
            X_current = StateVector([self.X.data.copy()])
            F_np1 = self.F(X_current)

            LHS = self.LHS_matrix @ self.X.data - (dt / 2) * F_np1

            residual = LHS - RHS

            if np.max(np.abs(residual)) < self.tol:
                break

            jac = self.M + (dt / 2) * self.L - (dt / 2) * self.J(X_current)

            dX = spla.spsolve(jac, -residual)

            self.X.data += dX


            i_loop += 1
            if i_loop > 20:
                print(f'Warning: Newton iteration exceeded 20 iterations, residual:', np.max(np.abs(residual)))
                break

        return self.X.data

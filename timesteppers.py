import numpy as np


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
        else:
            raise ValueError("Adams-Bashforth of order higher than 4 is not implemented")

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

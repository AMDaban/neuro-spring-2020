import math

from neurokit.learning_rule import STDP, RMSTDP


class Synapse:
    def __init__(self, src, dest, context, w=0, d=1):
        self.src = src
        self.dest = dest
        self.context = context
        self.w = w
        self.d = d

        # d must be greater than 0
        if self.d <= 0:
            self.d = 1

        self.s_times = []
        self.d_times = []

    def register_spike(self):
        t, dt = self.context.t(), self.context.dt()

        self.s_times.append(t)
        self.dest.register_potential_change(self.w, t + self.d * dt)

        self._apply_learning_rule()

    def notify_spike(self):
        t = self.context.t()

        self.d_times.append(t)

        self._apply_learning_rule()

    def steps(self, n):
        for _ in range(n):
            self._step()

    def _apply_learning_rule(self):
        learning_rule = self.context.learning_rule()
        if isinstance(learning_rule, STDP):
            self._apply_stdp()
        elif isinstance(learning_rule, RMSTDP):
            self._apply_rmstdp()

    def _apply_stdp(self):
        rule = self.context.learning_rule()
        a_p, a_n, tau_p, tau_n = rule.a_p, rule.a_n, rule.tau_p, rule.tau_n

        if (len(self.s_times) == 0) or (len(self.d_times) == 0):
            return

        last_s = self.s_times[-1]
        last_d = self.d_times[-1]

        a, tau, delta_t = a_p, tau_p, math.fabs(last_s - last_d)
        if last_d < last_s:
            a, tau = a_n, tau_n

        if delta_t == 0:
            return

        w_change = a * math.exp(-1 * delta_t / tau)

        self.w += w_change

    def _apply_rmstdp(self):
        pass

    def _step(self):
        pass

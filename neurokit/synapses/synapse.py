import math

from neurokit.learning_rule import STDP, RMSTDP


class Synapse:
    def __init__(self, src, dest, context, w=0, d=1):
        self.src = src
        self.dest = dest
        self.context = context
        self.w = float(w)
        self.d = float(d)

        # d must be greater than 0
        if self.d <= 0:
            self.d = 1

        # RMSTDP variables
        self.rmstdp_c = 0.0
        self.rmstdp_c_change = 0.0
        self.rmstdp_s = float(w)
        self.rmstdp_d = 0.0

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

    def step(self):
        learning_rule = self.context.learning_rule()
        if isinstance(learning_rule, RMSTDP):
            self._update_rmstdp_c()
            self._update_rmstdp_d()
            self._update_rmstdp_s()
            self.w = self.rmstdp_s

        self.rmstdp_c_change = 0

    def _update_rmstdp_s(self):
        dt = self.context.dt()
        self.rmstdp_s += (self.rmstdp_c * self.rmstdp_d) * dt

    def _update_rmstdp_d(self):
        learning_rule = self.context.learning_rule()
        dt, dopamine = self.context.dt(), self.context.dopamine()
        d, tau_d = self.rmstdp_d, learning_rule.tau_d
        dd_dt = -1 * (d / tau_d) + dopamine
        self.rmstdp_d += dd_dt * dt

    def _update_rmstdp_c(self):
        learning_rule = self.context.learning_rule()
        dt, tau_c = self.context.dt(), learning_rule.tau_c
        c, c_change = self.rmstdp_c, self.rmstdp_c_change
        dc_dt = -1 * (c / tau_c) + c_change
        self.rmstdp_c += dc_dt * dt

    def _apply_learning_rule(self):
        learning_rule = self.context.learning_rule()
        if isinstance(learning_rule, STDP):
            self._apply_stdp()
        elif isinstance(learning_rule, RMSTDP):
            self._apply_rmstdp()

    def _apply_stdp(self):
        rule = self.context.learning_rule()
        w_change = self._stdp_value(rule)
        self.w += w_change

    def _stdp_value(self, stdp_rule):
        a_p, a_n, tau_p, tau_n = stdp_rule.a_p, stdp_rule.a_n, stdp_rule.tau_p, stdp_rule.tau_n

        if (len(self.s_times) == 0) or (len(self.d_times) == 0):
            return 0

        last_s = self.s_times[-1]
        last_d = self.d_times[-1]

        a, tau, delta_t = a_p, tau_p, math.fabs(last_s - last_d)
        if last_d < last_s:
            a, tau = a_n, tau_n

        return a * math.exp(-1 * delta_t / tau)

    def _apply_rmstdp(self):
        rule = self.context.learning_rule()
        self.rmstdp_c_change += self._stdp_value(rule.stdp_rule)

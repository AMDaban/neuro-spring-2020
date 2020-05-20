import math


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
        # print("SPIKE", self.src.name, self.context.t(), self.w)
        t, dt = self.context.t(), self.context.dt()

        self.s_times.append(t)
        self.dest.register_potential_change(self.w, t + self.d * dt)

        self._apply_stdp()

    def notify_spike(self):
        print("OUT_SPIKE", self.dest.name, self.context.t())
        t = self.context.t()

        self.d_times.append(t)

        self._apply_stdp()

    def _apply_stdp(self):
        enabled, a_p, a_n, tau_p, tau_n = self.context.stdp_info()
        if not enabled:
            return

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

        # print(w_change, delta_t, self.src.name, self.dest.name)

        self.w += w_change

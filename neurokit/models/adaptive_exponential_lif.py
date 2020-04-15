import math

from neurokit.models.exceptions import InvalidTimeDelta, InvalidObserve
from neurokit.monitors.neuron_monitor import NeuronMonitor


class AdaptiveExponentialLIF:
    # TODO: tune default parameters
    def __init__(self, c_func, tau_m=20, tau_w=20, u_r=-80, r=10, u_t=0, delta_t=1, theta_rh=1, a=1, b=1, dt=0.001):
        """
        Exponential Leaky Integrate and Fire neuron model

        :param c_func:      current function, i.e. I(t)
        :param tau_m:       time constant (in u ode)
        :param tau_w:       time constant (i w ode)
        :param u_r:         rest potential
        :param r:           resistance
        :param u_t:         threshold potential
        :param dt:          time window in milliseconds
        :param delta_t:     sharpness parameter
        :param theta_rh:    firing threshold
        :param a:           source of subthreshold adaptation
        :param b:           spike coefficient
        :param dt:          time window size
        """
        self.c_func = c_func
        self.tau_m = float(tau_m)
        self.tau_w = float(tau_w)
        self.u_r = float(u_r)
        self.r = float(r)
        self.u_t = float(u_t)
        self.delta_t = float(delta_t)
        self.theta_rh = float(theta_rh)
        self.a = float(a)
        self.b = float(b)
        self.dt = float(dt)
        self.observe = True

        # current potential
        self._u = self.u_r

        # abstract current variables
        self._w = 0

        # needed to compute w
        self._spike_count = 0

        # current time
        self._t = 0.0

        # monitor
        self._monitor = NeuronMonitor()

    def set_observe(self, observe):
        if not isinstance(observe, bool):
            raise InvalidObserve()
        self.observe = observe

    def get_monitor(self):
        return self._monitor

    def _comp_dw(self):
        """
        Computes current dw

        :return: dw
        """
        a, b = self.a, self.b
        dt = self.dt
        u, u_r = self._u, self.u_r
        w = self._w
        spike_count, tau_w = self._spike_count, self.tau_w

        dw_dt = (a * (u - u_r) - w + b * tau_w * spike_count) / tau_w
        return dw_dt * dt

    def _comp_du(self):
        """
        Computes current du

        :return: du
        """
        u, u_r = self._u, self.u_r
        t, dt, tau_m = self._t, self.dt, self.tau_m
        delta_t, theta_rh = self.delta_t, self.theta_rh
        r, w = self.r, self._w
        c_func = self._c_func

        f_u = -(u - u_r) + delta_t * math.exp((u - theta_rh) / delta_t)
        du_dt = (f_u - r * w + r * c_func(t)) / tau_m
        return du_dt * dt

    def _step(self):
        """
        Simulates next state of model and set variables,
        also this method observe u values and spike times
        """
        self._w = self._w + self._comp_dw()

        next_u = self._u + self._comp_du()
        next_t = self._t + self.dt

        spiked = False
        if next_u >= self.u_t:
            self._spike_count += 1
            next_u = self.u_r
            spiked = True

        if self.observe:
            self._monitor.observe(next_t, next_u, self._c_func(self._t), spiked)

        self._u = next_u
        self._t = next_t

    def steps(self, n):
        """
        Simulate next n steps of model

        :param n: number of steps to simulate
        """
        if (self._t == 0.0) and self.observe:
            self._monitor.observe(self._t, self._u, self._c_func(self._t), False)

        for _ in range(n):
            self._step()

    def _c_func(self, t):
        return float(self.c_func(t))

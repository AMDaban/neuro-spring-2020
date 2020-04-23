import math

from neurokit.models.lif import LIF
from neurokit.models.exceptions import InvalidTimeDelta, InvalidObserve
from neurokit.monitors.neuron_monitor import NeuronMonitor


class AdaptiveExponentialLIF(LIF):
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
        :param delta_t:     sharpness parameter
        :param theta_rh:    firing threshold
        :param a:           source of subthreshold adaptation
        :param b:           spike coefficient
        :param dt:          time window size
        """

        LIF.__init__(self, c_func=c_func, u_r=u_r, r=r, u_t=u_t, dt=dt)

        self.tau_m = float(tau_m)
        self.tau_w = float(tau_w)
        self.delta_t = float(delta_t)
        self.theta_rh = float(theta_rh)
        self.a = float(a)
        self.b = float(b)

        self._w = 0

    def _comp_dw(self):
        """
        Computes current dw

        :return: dw
        """
        a, b = self.a, self.b
        dt, tau_w = self.dt, self.tau_w
        u, u_r = self._u, self.u_r
        w = self._w
        _, _, _, spikes = self.get_monitor().get_observations()

        dw_dt = (a * (u - u_r) - w + b * tau_w * len(spikes)) / tau_w
        return dw_dt * dt

    # noinspection DuplicatedCode
    def _comp_du(self):
        """
        Computes current du, also updates w

        :return: du
        """
        self._w = self._w + self._comp_dw()

        u, u_r = self._u, self.u_r
        t, dt, tau_m = self._t, self.dt, self.tau_m
        delta_t, theta_rh = self.delta_t, self.theta_rh
        r, w = self.r, self._w
        c_func = self._c_func

        f_u = -(u - u_r) + delta_t * math.exp((u - theta_rh) / delta_t)
        du_dt = (f_u - r * w + r * c_func(t)) / tau_m
        return du_dt * dt

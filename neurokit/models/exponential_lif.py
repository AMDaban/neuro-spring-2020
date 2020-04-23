import math

from neurokit.models.lif import LIF
from neurokit.models.exceptions import InvalidTimeDelta, InvalidObserve
from neurokit.monitors.neuron_monitor import NeuronMonitor


class ExponentialLIF(LIF):
    # TODO: tune default parameters
    def __init__(self, c_func, tau=20, u_r=-80, r=10, u_t=0, delta_t=1, theta_rh=1, dt=0.001):
        """
        Exponential Leaky Integrate and Fire neuron model

        :param c_func:      current function, i.e. I(t)
        :param tau:         time constant
        :param u_r:         rest potential
        :param r:           resistance
        :param u_t:         threshold potential
        :param delta_t:     sharpness parameter
        :param theta_rh:    firing threshold
        :param dt:          time window size
        """

        LIF.__init__(self, c_func=c_func, tau=tau, u_r=u_r, r=r, u_t=u_t, dt=dt)

        self.delta_t = float(delta_t)
        self.theta_rh = float(theta_rh)

    # noinspection DuplicatedCode
    def _comp_du(self):
        """
        Computes current du

        :return: du
        """
        u, u_r = self._u, self.u_r
        t, dt, tau = self._t, self.dt, self.tau
        delta_t, theta_rh = self.delta_t, self.theta_rh
        r = self.r
        c_func = self._c_func

        f_u = -(u - u_r) + delta_t * math.exp((u - theta_rh) / delta_t)
        du_dt = (f_u + r * c_func(t)) / tau
        return du_dt * dt

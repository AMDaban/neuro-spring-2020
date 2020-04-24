import math

from neurokit.models.lif import LIF
from neurokit.monitors.neuron_monitor import NeuronMonitor
from neurokit.context import Context


class ExponentialLIF(LIF):
    # TODO: tune default parameters
    def __init__(self, tau=20, u_r=-80, r=10, u_t=0, delta_t=1, theta_rh=1, context=Context(0.001)):
        """
        Exponential Leaky Integrate and Fire neuron model

        :param tau:         time constant
        :param u_r:         rest potential
        :param r:           resistance
        :param u_t:         threshold potential
        :param delta_t:     sharpness parameter
        :param theta_rh:    firing threshold
        :param context:     global context
        """

        LIF.__init__(self, tau=tau, u_r=u_r, r=r, u_t=u_t, context=context)

        self.delta_t = float(delta_t)
        self.theta_rh = float(theta_rh)

    # noinspection DuplicatedCode
    def _comp_du(self):
        """
        Computes current du

        :return: du
        """
        u, u_r = self._u, self.u_r
        dt, tau = self.context.dt(), self.tau
        delta_t, theta_rh = self.delta_t, self.theta_rh
        r, c = self.r, self._c

        f_u = -(u - u_r) + delta_t * math.exp((u - theta_rh) / delta_t)
        du_dt = (f_u + r * c) / tau
        return du_dt * dt

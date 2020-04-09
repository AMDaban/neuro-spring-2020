from scipy.integrate import odeint

from neurokit.models.neuron_state import NeuronState


class LIF:
    def __init__(self, c_func, tau=1, u_r=-80, r=10, u_t=0):
        """
        :param tau:    time constant
        :param u_r:    rest potential
        :param r:      resistance
        :param u_t:    threshold potential
        """
        self.tau = tau
        self.u_r = u_r
        self.r = r
        self.u_t = u_t
        self.c_func = c_func

        # current potential
        self._u = u_r

        # current time
        self._t = 0

    def run(self, delta_t):
        """
        calculate next state of neuron at t + delta_t and return it.

        :return: next state of neuron at t + delta_t
        """

        def du_dt(u, t):
            return (-(u - self.u_r) + (self.r * self.c_func(t))) / self.tau

        next_values = odeint(du_dt, self._u, [self._t, self._t + delta_t])

        self._u = next_values[-1]
        self._t += delta_t

        return NeuronState(self._u, self._t)

from scipy.integrate import odeint

from neurokit.models.exceptions import InvalidTimeDelta


class LIF:
    # TODO: tune default parameters
    def __init__(self, c_func, tau=1, u_r=-80, r=10, u_t=0):
        """
        Leaky Integrate and Fire neuron model

        :param c_func: current function, i.e. I(t)
        :param tau:    time constant
        :param u_r:    rest potential
        :param r:      resistance
        :param u_t:    threshold potential
        :param dt:     time window in milliseconds
        """
        self.c_func = c_func
        self.tau = tau
        self.u_r = u_r
        self.r = r
        self.u_t = u_t

        # current potential
        self._u = u_r

        # current time
        self._t = 0

    def simulate(self, delta_t):
        """
        simulate neuron for next delta_t milliseconds

        :return: u_values, t_values, spike_times
        """

        if (delta_t < 0) or (not isinstance(delta_t, int)):
            raise InvalidTimeDelta()

        spike_times = []
        u_values = []
        t_values = []

        for t in range(self._t, self._t + delta_t):
            next_values = odeint(self._du_dt, self._u, [t, t + 1])
            self._u = next_values[-1]

            u_values.append(self._u)
            t_values.append(t)
            if self._u >= self.u_t:
                self._u = self.u_r
                spike_times.append(t)

        self._t += delta_t

        return u_values, t_values, spike_times

    def _du_dt(self, u, t):
        return (-(u - self.u_r) + (self.r * self.c_func(t))) / self.tau

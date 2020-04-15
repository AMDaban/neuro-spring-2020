from neurokit.models.exceptions import InvalidTimeDelta, InvalidObserve
from neurokit.monitors.neuron_monitor import NeuronMonitor


class LIF:
    # TODO: tune default parameters
    def __init__(self, c_func, tau=20, u_r=-80, r=10, u_t=0, dt=0.001):
        """
        Exponential Leaky Integrate and Fire neuron model

        :param c_func:      current function, i.e. I(t)
        :param tau:         time constant
        :param u_r:         rest potential
        :param r:           resistance
        :param u_t:         threshold potential
        :param dt:          time window in milliseconds
        """
        self.c_func = c_func
        self.tau = float(tau)
        self.u_r = float(u_r)
        self.r = float(r)
        self.u_t = float(u_t)
        self.dt = float(dt)
        self.observe = True

        # current potential
        self._u = self.u_r

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

    def _comp_du(self):
        """
        Computes current du

        :return: du
        """
        u, u_r = self._u, self.u_r
        t, dt, tau = self._t, self.dt, self.tau
        r = self.r
        c_func = self._c_func

        f_u = -(u - u_r)
        du_dt = (f_u + r * c_func(t)) / tau
        return du_dt * dt

    def _step(self):
        """
        Simulates next state of model and set variables,
        also this method observe u values and spike times
        """
        next_u = self._u + self._comp_du()
        next_t = self._t + self.dt

        spiked = False
        if next_u >= self.u_t:
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

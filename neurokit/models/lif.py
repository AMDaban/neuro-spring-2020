from collections import defaultdict

from neurokit.monitors.neuron_monitor import NeuronMonitor
from neurokit.synapses.synapse import Synapse
from neurokit.context import Context


class LIF:
    # TODO: tune default parameters
    def __init__(self, tau=20, u_r=-80, r=10, u_t=0, context=Context(0.001), name=""):
        """
        Exponential Leaky Integrate and Fire neuron model

        :param tau:         time constant
        :param u_r:         rest potential
        :param r:           resistance
        :param u_t:         threshold potential
        :param context:     global context
        """

        def default_in_c(t):
            return 0

        self.tau = float(tau)
        self.u_r = float(u_r)
        self.r = float(r)
        self.u_t = float(u_t)
        self.context = context
        self.name = name

        self._c = 0.0
        self._u = self.u_r
        self._monitor = NeuronMonitor()
        self._out_synapses = []
        self._in_synapses = []
        self._potential_changes = defaultdict(float)
        self._in_c = default_in_c

    def register_out_synapse(self, synapse):
        """
        Register out synapse

        :param synapse: an instance of Synapse class
        """
        self._out_synapses.append(synapse)

    def register_in_synapse(self, synapse):
        """
        Register in synapse

        :param synapse: an instance of Synapse class
        """
        self._in_synapses.append(synapse)

    def register_potential_change(self, dw, t):
        if t > self.context.t():
            self._potential_changes[t] += dw

    def get_monitor(self):
        """
        Get neuron monitor

        :return: neuron monitor
        """
        return self._monitor

    def set_in_c(self, c_func):
        self._in_c = c_func

    def steps(self, n):
        """
        Simulate next n steps of model

        :param n: number of steps to simulate
        """
        t = self.context.t()

        if t == 0.0:
            self._monitor.observe(t, self._u, self._c, False)

        for _ in range(n):
            self._step()

    def _step(self):
        """
        Simulates next state of model and set variables,
        also this method observe u values and spike times
        """
        t, dt = self.context.t(), self.context.dt()

        pre_synaptic_change = self._potential_changes.pop(t, 0)

        self._update_c()

        next_u = self._u + self._comp_du() + pre_synaptic_change
        next_t = t + dt

        spiked = False
        if next_u >= self.u_t:
            next_u = self.u_r
            spiked = True

        self._monitor.observe(next_t, next_u, self._c, spiked)

        if spiked:
            for out_synapse in self._out_synapses:
                out_synapse.register_spike()
            for in_synapse in self._in_synapses:
                in_synapse.notify_spike()

        self._u = next_u

    def _update_c(self):
        t = self.context.t()
        self._c = float(self._in_c(t))

    def _comp_du(self):
        """
        Computes current du

        :return: du
        """
        u, u_r = self._u, self.u_r
        dt, tau = self.context.dt(), self.tau
        r, c = self.r, self._c

        f_u = -(u - u_r)
        du_dt = (f_u + r * c) / tau
        return du_dt * dt

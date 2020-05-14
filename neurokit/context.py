class Context:
    def __init__(self, dt, stdp_enabled=True, a_p=1, a_n=1, tau_p=1, tau_n=1):
        """
        Neural Context Contains Global Configurations of neurons and synapses
        """
        self._dt = float(dt)
        self.stdp_enabled = stdp_enabled
        self.a_p = float(a_p)
        self.a_n = float(a_n)
        self.tau_p = float(tau_p)
        self.tau_n = float(tau_n)

        self._t = 0.0

    def t(self):
        """
        Returns current time

        :return: current time
        """
        return self._t

    def dt(self):
        """
        Returns dt

        :return: dt
        """
        return self._dt

    def step(self):
        """
        Compute next state of Context
        """
        self._t = self.t() + self.dt()

    def stdp_info(self):
        """
        Returns stdp related information

        :return: stdp related information
        """

        return self.stdp_enabled, self.a_p, self.a_n, self.tau_p, self.tau_n

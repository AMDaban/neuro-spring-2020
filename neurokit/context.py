class Context:
    def __init__(self, dt, learning_rule=None):
        """
        Neural Context Contains Global Configurations of neurons and synapses
        """
        self._dt = float(dt)
        self._learning_rule = learning_rule

        self._t = 0.0
        self._dop = 0.0

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
        self._dop = 0.0

    def learning_rule(self):
        """
        Returns learning rule
        :return: learning rule
        """
        return self._learning_rule

    def dopamine(self):
        """
        Returns dopamine
        :return: dopamine
        """
        return self._dop

    def set_dopamine(self, dop):
        """
        sets dopamine
        :param dop: value to set
        """
        self._dop = dop

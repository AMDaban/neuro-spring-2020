class Context:
    def __init__(self, dt):
        """
        Neural Context Contains Global Configurations of neurons and synapses
        """
        self.dt = float(dt)

        self._t = 0

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
        return self.dt

    def step(self):
        """
        Compute next state of Context
        """
        self._t += self.dt

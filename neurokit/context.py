class Context:
    def __init__(self, dt, learning_rule=None):
        """
        Neural Context Contains Global Configurations of neurons and synapses
        """
        self._dt = float(dt)
        self._learning_rule = learning_rule

        self._t = 0.0
        self._dop_schedules = []

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
        dop = 0
        dop_schedules, t = self._dop_schedules, self.t()

        for schedule in self._dop_schedules:
            if schedule[0] <= t < schedule[1]:
                dop += schedule[2]

        return dop

    def change_dopamine(self, dop_delta, steps):
        """
        change dopamine
        :param dop_delta: value to change
        :param steps: steps count
        """
        next_t = self.t() + 1
        self._dop_schedules.append((next_t, next_t + steps, dop_delta))

from neurokit.monitors.exceptions import InvalidObservations


class NeuronMonitor:
    def __init__(self):
        self._times = []
        self._u_values = []
        self._c_values = []
        self._spikes = []

    def _check_values(self, t, u, c, spiked):
        if (not isinstance(t, float)) and t < 0:
            raise InvalidObservations()
        last_t = self._times[-1] if len(self._times) > 0 else -1
        if t <= last_t:
            raise InvalidObservations()

        if not isinstance(u, float):
            raise InvalidObservations()

        if not isinstance(c, float):
            raise InvalidObservations()

        if not isinstance(spiked, bool):
            raise InvalidObservations()

    def observe(self, t, u, c, spiked):
        self._check_values(t, u, c, spiked)

        self._times.append(t)
        self._u_values.append(u)
        self._c_values.append(c)
        if spiked:
            self._spikes.append(t)

    def get_observations(self):
        return self._times, self._u_values, self._c_values, self._spikes

    def last_observation(self):
        if len(self._times) == 0:
            return None

        last_t = self._times[-1]
        last_u = self._u_values[-1]
        last_c = self._c_values[-1]
        spiked = (len(self._spikes) != 0) and (self._spikes[-1] == last_t)

        return last_t, last_u, last_c, spiked

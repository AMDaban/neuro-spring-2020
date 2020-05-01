class PopulationMonitor:
    def __init__(self):
        self._spikes = []

    def observe(self, t, spiked_indices):
        self._spikes.append((t, spiked_indices))

    def get_observations(self):
        return self._spikes

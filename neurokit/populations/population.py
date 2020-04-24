from neurokit.populations.exceptions import InvalidIndex
from neurokit.synapses.synapse import Synapse
from neurokit.monitors.population_monitor import PopulationMonitor


class Population:
    def __init__(self, identifier, dimensions, context, neuron_init):
        self.name = identifier
        self.x_dim, self.y_dim = dimensions
        self.context = context

        self._monitor = PopulationMonitor()

        self.neurons = []
        for i in range(self.x_dim):
            row = []
            for j in range(self.y_dim):
                row.append(neuron_init(i, j))
            self.neurons.append(row)

    def connect_two(self, src, dest, w, d):
        try:
            src_neuron = self.get_neuron(src[0], src[1])
            dest_neuron = self.get_neuron(dest[0], dest[1])
        except IndexError:
            raise InvalidIndex()

        if (src is None) or (dest is None):
            raise InvalidIndex()

        synapse = Synapse(src_neuron, dest_neuron, self.context, w, d)
        src_neuron.register_out_synapse(synapse)

    def get_neuron(self, i, j):
        try:
            return self.neurons[i][j]
        except IndexError:
            return None

    def steps(self, n):
        for i in range(n):
            self._step()

    def set_neuron_in_c(self, idx, c):
        neuron = self.get_neuron(idx[0], idx[1])
        if neuron is None:
            raise InvalidIndex()

        neuron.set_in_c(c)

    def set_pop_in_c(self, c):
        for neuron in self._neuron_iter():
            neuron.set_in_c(c)

    def _step(self):
        current_t = self.context.t()
        spiked_indices = []

        for i in range(len(self.neurons)):
            for j in range(len(self.neurons[i])):
                self.neurons[i][j].steps(1)

                _, _, _, spiked = self.neurons[i][j].last_observation()
                if spiked:
                    spiked_indices.append((i, j))

        if len(spiked_indices) > 0:
            self._monitor.observe(current_t, spiked_indices)

    def _neuron_iter(self):
        for i in range(len(self.neurons)):
            for j in range(len(self.neurons[i])):
                yield self.neurons[i][j]

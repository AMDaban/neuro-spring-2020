from neurokit.populations.exceptions import InvalidIndex
from neurokit.synapses.synapse import Synapse


class Population:
    def __init__(self, dimensions, context, neuron_init):
        self.x_dim, self.y_dim = dimensions
        self.context = context

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

    def _step(self):
        for neuron in self._neuron_iter():
            neuron.steps(1)

    def _neuron_iter(self):
        for i in range(len(self.neurons)):
            for j in range(len(self.neurons[i])):
                yield self.neurons[i][j]

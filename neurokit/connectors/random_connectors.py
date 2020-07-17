import random

from .base import Connector
from neurokit.synapses.synapse import Synapse

class RandomConnectorFixedProb(Connector):
    def __init__(self, pre, post, w_mu, w_sigma, w_mu_inh, w_sigma_inh, d, is_exc_func, prob):
        Connector.__init__(self, pre, post, w_mu, w_sigma, w_mu_inh, w_sigma_inh, d, is_exc_func)
        self._prob = prob

    def connect(self, context):
        pre_x, pre_y = self._pre.x_dim, self._pre.y_dim
        post_x, post_y = self._post.x_dim, self._post.y_dim
        self_connection = self._pre.name == self._post.name

        pre_size = pre_x * pre_y
        post_size = post_x * post_y

        if self_connection:
            possible_connections = pre_size * (post_size - 1)
        else:
            possible_connections = pre_size * post_size

        connections = possible_connections * self._prob

        connected_count = 0
        connected_neurons = set()
        while connected_count < connections:
            i_pre, j_pre = random.randint(0, pre_x - 1), random.randint(0, pre_y - 1)
            i_post, j_post = random.randint(0, post_x - 1), random.randint(0, post_y - 1)

            if self_connection and ((i_pre, j_pre,) == (i_post, j_post,)):
                continue

            key = ((i_pre, j_pre), (i_post, j_post),)
            if key in connected_neurons:
                continue
            connected_neurons.add(key)

            src_neuron = self._pre.get_neuron(i_pre, j_pre)
            dest_neuron = self._post.get_neuron(i_post, j_post)

            synapse = Synapse(src_neuron, dest_neuron, context, self._get_weight((i_pre, j_pre)), self._d)
            src_neuron.register_out_synapse(synapse)

            connected_count += 1


class RandomConnectorFixedPre(Connector):
    def __init__(self, pre, post, w_mu, w_sigma, w_mu_inh, w_sigma_inh, d, is_exc_func, pre_cons):
        Connector.__init__(self, pre, post, w_mu, w_sigma, w_mu_inh, w_sigma_inh, d, is_exc_func)
        self._pre_cons = pre_cons

    def connect(self, context):
        pre_x, pre_y = self._pre.x_dim, self._pre.y_dim
        post_x, post_y = self._post.x_dim, self._post.y_dim
        self_connection = self._pre.name == self._post.name

        for i_post in range(post_x):
            for j_post in range(post_y):
                connected_count = 0
                connected_neurons = set()
                while connected_count < self._pre_cons:
                    i_pre, j_pre = random.randint(0, pre_x - 1), random.randint(0, pre_y - 1)

                    if self_connection and ((i_pre, j_pre,) == (i_post, j_post,)):
                        continue

                    key = (i_pre, j_pre,)
                    if key in connected_neurons:
                        continue
                    connected_neurons.add(key)

                    src_neuron = self._pre.get_neuron(i_pre, j_pre)
                    dest_neuron = self._post.get_neuron(i_post, j_post)

                    synapse = Synapse(src_neuron, dest_neuron, context, self._get_weight((i_pre, j_pre)), self._d)
                    src_neuron.register_out_synapse(synapse)

                    connected_count += 1
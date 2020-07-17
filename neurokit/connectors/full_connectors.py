import numpy as np

from .base import Connector
from neurokit.synapses.synapse import Synapse

class FullConnector(Connector):
    def connect(self, context):
        pre_x, pre_y = self._pre.x_dim, self._pre.y_dim
        post_x, post_y = self._post.x_dim, self._post.y_dim
        self_connection = self._pre.name == self._post.name

        for i_pre in range(pre_x):
            for j_pre in range(pre_y):
                for i_post in range(post_x):
                    for j_post in range(post_y):
                        if self_connection and ((i_pre, j_pre,) == (i_post, j_post,)):
                            continue

                        src_neuron = self._pre.get_neuron(i_pre, j_pre)
                        dest_neuron = self._post.get_neuron(i_post, j_post)


                        synapse = Synapse(src_neuron, dest_neuron, context, self._get_weight((i_pre, j_pre)), self._d)
                        src_neuron.register_out_synapse(synapse)
                        dest_neuron.register_in_synapse(synapse)
        
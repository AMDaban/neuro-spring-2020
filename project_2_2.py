import random

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from neurokit.context import Context
from neurokit.populations.population import Population
from neurokit.models.lif import LIF
from neurokit.synapses.synapse import Synapse

x_size = 10
y_size = 10
n = x_size * y_size
simulation_steps = 10000
dt = 0.001

first_non_zero_c_time = 0.005
last_non_zero_c_time = 0.09
random_c_func_maximum_c_change = 0.3
random_c_func_maximum_c = 1
c_samples = {}

inner_connections = 100
e_outer_connections = 100
i_outer_connections = 100
w_ee = 10
w_ie = -1000
w_ei = 10
d_con = 1

tau = 10
u_r = -70
u_t = -50
r = 10


def get_pop_in_c():
    c = 0

    def pop_in_c(t):
        nonlocal c

        if c_samples.get(t) is not None:
            return c_samples.get(t)

        if t < first_non_zero_c_time or t > last_non_zero_c_time:
            c = 0
        else:
            c += (random.random() - 0.5) * random_c_func_maximum_c_change
            if c < 0:
                c = 0
            if c > random_c_func_maximum_c:
                c = random_c_func_maximum_c

        c_samples[t] = c

        return c

    return pop_in_c


def get_neuron_init(context):
    def neuron_init(x, y):
        return LIF(context=context, tau=tau, u_r=u_r, u_t=u_t, r=r)

    return neuron_init


def connect_inner_neurons(population):
    connected_indices = set()
    connected_neurons = 0
    while connected_neurons < inner_connections:
        first_x, first_y = random.randint(0, x_size - 1), random.randint(0, y_size - 1)
        sec_x, sec_y = random.randint(0, x_size - 1), random.randint(0, y_size - 1)

        if (first_x, first_y) == (sec_x, sec_y):
            continue

        key = f"{first_x}_{first_y}_{sec_x}_{sec_y}"
        if key in connected_indices:
            continue
        connected_indices.add(key)

        population.connect_two((first_x, first_y), (sec_x, sec_y), w_ee, d_con)
        connected_neurons += 1


def connect_outer_neurons(p1, p2, mode, context):
    connected_indices = set()
    connected_neurons = 0

    outer_connections = e_outer_connections if mode == 'ei' else i_outer_connections

    while connected_neurons < outer_connections:
        first_x, first_y = random.randint(0, x_size - 1), random.randint(0, y_size - 1)
        sec_x, sec_y = random.randint(0, x_size - 1), random.randint(0, y_size - 1)

        key = f"{first_x}_{first_y}_{sec_x}_{sec_y}"
        if key in connected_indices:
            continue
        connected_indices.add(key)

        src_neuron = p1.get_neuron(first_x, first_y)
        dest_neuron = p2.get_neuron(sec_x, sec_y)

        w = w_ei if mode == 'ei' else w_ie

        synapse = Synapse(src_neuron, dest_neuron, context, w, d_con)
        src_neuron.register_out_synapse(synapse)

        connected_neurons += 1


def plot_result(population):
    spikes = population.get_monitor().get_observations()

    spike_count = 0

    data_source = {'neuron_idx': [], 'time': [], 'group': []}
    for t_spikes in spikes:
        t = t_spikes[0]
        spiked_neurons = t_spikes[1]
        for spiked_neuron in spiked_neurons:
            spike_count += 1

            neuron_idx = spiked_neuron[0] * y_size + spiked_neuron[1]
            data_source['neuron_idx'].append(neuron_idx)
            data_source['time'].append(t)
            data_source['group'].append("ex")

    print(population.name, spike_count / (dt * simulation_steps))

    df = pd.DataFrame(data=data_source)

    sns.scatterplot(x="time", y="neuron_idx", data=df, hue="group", s=5)
    plt.title("Raster Plot")
    plt.ylabel("Neuron")
    plt.xlabel("time")
    plt.show()


def main():
    context = Context(dt=dt)

    pop_in_c = get_pop_in_c()

    e1 = Population("e1", (x_size, y_size), context, get_neuron_init(context))
    e1.set_pop_in_c(pop_in_c)
    connect_inner_neurons(e1)

    e2 = Population("e2", (x_size, y_size), context, get_neuron_init(context))
    e2.set_pop_in_c(pop_in_c)
    connect_inner_neurons(e2)

    i1 = Population("i1", (x_size, y_size), context, get_neuron_init(context))

    connect_outer_neurons(e1, i1, "ei", context)
    connect_outer_neurons(e2, i1, "ei", context)
    connect_outer_neurons(i1, e1, "ie", context)
    connect_outer_neurons(i1, e2, "ie", context)

    for i in range(simulation_steps):
        e1.steps(1)
        e2.steps(1)
        i1.steps(1)
        context.step()

    plot_result(e1)
    plot_result(e2)
    plot_result(i1)


if __name__ == '__main__':
    main()

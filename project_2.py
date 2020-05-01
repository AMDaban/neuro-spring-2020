import random

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from neurokit.context import Context
from neurokit.populations.population import Population
from neurokit.models.lif import LIF

x_size = 5
y_size = 100
n = x_size * y_size
exc_size = (4 * n) / 5
simulation_steps = 1000
dt = 0.001

first_non_zero_c_time = 0.005
last_non_zero_c_time = 0.09
random_c_func_maximum_c_change = 0.3
random_c_func_maximum_c = 1
c_samples = {}

connections = 1000
w_con = 10
w_in_con = -20
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


def connect_neurons(population):
    connected_indices = set()
    connected_neurons = 0
    while connected_neurons < connections:
        first_x, first_y = random.randint(0, x_size - 1), random.randint(0, y_size - 1)
        sec_x, sec_y = random.randint(0, x_size - 1), random.randint(0, y_size - 1)

        key = f"{first_x}_{first_y}_{sec_x}_{sec_y}"
        if key in connected_indices:
            continue
        connected_indices.add(key)

        if first_x * x_size + first_y <= exc_size:
            w = w_con
        else:
            w = w_in_con

        population.connect_two((first_x, first_y), (sec_x, sec_y), w, d_con)
        connected_neurons += 1


def plot_result(population):
    spikes = population.get_monitor().get_observations()

    data_source = {'neuron_idx': [], 'time': [], 'group': []}
    for t_spikes in spikes:
        t = t_spikes[0]
        spiked_neurons = t_spikes[1]
        for spiked_neuron in spiked_neurons:
            print("shit")
            neuron_idx = spiked_neuron[0] * y_size + spiked_neuron[1]
            data_source['neuron_idx'].append(neuron_idx)
            data_source['time'].append(t)
            if neuron_idx <= exc_size:
                data_source['group'].append("ex")
            else:
                data_source['group'].append("in")
    df = pd.DataFrame(data=data_source)

    print(df)

    sns.scatterplot(x="time", y="neuron_idx", data=df,  hue="group", s=5)
    plt.title("Raster Plot")
    plt.ylabel("Neuron")
    plt.xlabel("time")
    plt.show()

    # spike_chart = []
    # for t_spikes in spikes:
    #     t = t_spikes[0]
    #     spiked_neurons = t_spikes[1]
    #     for spiked_neuron in spiked_neurons:
    #         neuron_idx = spiked_neuron[0] * y_size + spiked_neuron[1]
    #         spike_chart.append((neuron_idx, t))
    #
    # spike_chart = np.array(spike_chart)
    #
    # print(len(c_samples))
    #
    # sns.scatterplot(x=spike_chart[:, 1], y=spike_chart[:, 0], s=5)
    # plt.title("Raster Plot")
    # plt.ylabel("Neuron")
    # plt.xlabel("time")
    # plt.show()


def main():
    context = Context(dt=dt)

    population = Population("main", (x_size, y_size), context, get_neuron_init(context))
    population.set_pop_in_c(get_pop_in_c())

    connect_neurons(population)

    for i in range(simulation_steps):
        population.steps(1)
        context.step()

    plot_result(population)


if __name__ == '__main__':
    main()

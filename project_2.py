import random

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from neurokit.context import Context
from neurokit.populations.population import Population
from neurokit.models.lif import LIF
from neurokit.connectors.full_connectors import FullConnector
from neurokit.connectors.random_connectors import RandomConnectorFixedPre, RandomConnectorFixedProb

x_size = 5
y_size = 200
n = x_size * y_size
exc_size = (4 * n) / 5
simulation_steps = 10000
dt = 0.001

first_non_zero_c_time = 0.5
last_non_zero_c_time = 9.5
random_c_func_maximum_c_change = 0.001
random_c_func_maximum_c = 10
initial_c = 2.5
c_samples = {}

connections=10000
w_con_mu = 5
w_con_sigma = 0
w_con_mu_inh = -1.5
w_con_sigma_inh = 0
d_con = 1

tau = 1
u_r = -70
u_t = -50
r = 10

def is_exc(src_idx):
    return (src_idx[0] * y_size + src_idx[0]) <= exc_size 

def get_pop_in_c():
    c = initial_c

    def pop_in_c(t):
        nonlocal c

        if c_samples.get(t) is not None:
            return c_samples.get(t)

        if t < first_non_zero_c_time or t > last_non_zero_c_time:
            c_samples[t] = 0
            return 0
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
        return LIF(context=context, tau=tau * (random.random() + 0.1), u_r=u_r, u_t=u_t * (random.random() + 0.1),
                   r=r * (random.random() + 0.1))

    return neuron_init


def plot_result(population):
    spikes = population.get_monitor().get_observations()

    data_source = {'neuron_idx': [], 'time': [], 'group': []}
    for t_spikes in spikes:
        t = t_spikes[0]
        spiked_neurons = t_spikes[1]
        for spiked_neuron in spiked_neurons:
            neuron_idx = spiked_neuron[0] * y_size + spiked_neuron[1]
            data_source['neuron_idx'].append(neuron_idx)
            data_source['time'].append(t)
            if neuron_idx <= exc_size:
                data_source['group'].append("ex")
            else:
                data_source['group'].append("in")
    df = pd.DataFrame(data=data_source)

    sns.scatterplot(x="time", y="neuron_idx", data=df, hue="group", s=5)
    plt.title("Raster Plot")
    plt.ylabel("Neuron")
    plt.xlabel("time")
    plt.show()

    def sort_func(x):
        return x[0]

    c_items = list(c_samples.items())
    c_items.sort(key=sort_func)
    ts = [x[0] for x in c_items]
    cs = [x[1] for x in c_items]
    plt.plot(ts, cs)
    plt.xlabel('time')
    plt.ylabel('C(t)')
    plt.show()


def main():
    context = Context(dt=dt)

    population = Population("main", (x_size, y_size), context, get_neuron_init(context))
    population.set_pop_in_c(get_pop_in_c())

    connector = RandomConnectorFixedPre(population, population, w_con_mu, w_con_sigma, w_con_mu_inh, w_con_sigma_inh, d_con, is_exc, 15)
    connector.connect(context)

    # connector = RandomConnectorFixedProb(population, population, w_con_mu, w_con_sigma, w_con_mu_inh, w_con_sigma_inh, d_con, is_exc, 0.01)
    # connector.connect(context)

    print("simulating...")

    for i in range(simulation_steps):
        population.steps(1)
        context.step()

    plot_result(population)


if __name__ == '__main__':
    main()

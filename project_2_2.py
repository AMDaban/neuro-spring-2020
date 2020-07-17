import random

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from neurokit.context import Context
from neurokit.populations.population import Population
from neurokit.models.lif import LIF
from neurokit.synapses.synapse import Synapse
from neurokit.connectors.full_connectors import FullConnector
from neurokit.connectors.random_connectors import RandomConnectorFixedPre, RandomConnectorFixedProb

exc_x_size = 4
exc_y_size = 100
exc_n = exc_x_size * exc_y_size
inh_x_size = 2
inh_y_size = 100
inh_n = inh_x_size * inh_x_size

simulation_steps = 1000
dt = 0.001

first_non_zero_c_time = 0.05
last_non_zero_c_time = 0.95
random_c_func_maximum_c_change = 0.1
random_c_func_maximum_c = 25.5
random_c_func_minimum_c = 24.5
initial_c = 25

c_samples_exc_1 = {}
c_samples_exc_2 = {}

exc_tau = 2
exc_u_r = -70
exc_u_t = -60
exc_r = 4

inh_tau = 5
inh_u_r = -70
inh_u_t = -60
inh_r = 8

w_ee = 10
w_ie = -1000
w_ei = 1000
d_con = 1

context = None

def get_pop_in_c(exc_num):
    c = initial_c

    if exc_num == 1:
        c_samples = c_samples_exc_1
    else:
        c_samples = c_samples_exc_2

    def pop_in_c(t):
        nonlocal c, c_samples

        if c_samples.get(t) is not None:
            return c_samples.get(t)

        if t < first_non_zero_c_time or t > last_non_zero_c_time:
            c_samples[t] = 0
            return 0
        else:
            c += (random.random() - 0.5) * random_c_func_maximum_c_change
            if c < random_c_func_minimum_c:
                c = random_c_func_minimum_c
            if c > random_c_func_maximum_c:
                c = random_c_func_maximum_c

        c_samples[t] = c

        return c

    return pop_in_c

def get_neuron_init(context, is_exc=True):
    def neuron_init(x, y):
        if is_exc:
            return LIF(context=context, tau=exc_tau, u_r=exc_u_r, u_t=exc_u_t, r=exc_r)
        else:
            return LIF(context=context, tau=inh_tau, u_r=inh_u_r, u_t=inh_u_t, r=inh_r)

    return neuron_init

def plot_res(e1, e2, i1):
    spikes = e1.get_monitor().get_observations()
    e1a = [len(x[1])/exc_n for x in spikes]
    t = [x[0] for x in spikes]

    spikes = e2.get_monitor().get_observations()
    e2a = [len(x[1])/exc_n for x in spikes]

    spikes = i1.get_monitor().get_observations()
    i1a = [len(x[1])/inh_n for x in spikes]

    plt.plot(t, e1a, label="P1")
    plt.plot(t, e2a, label="P2")
    # plt.plot(t, i1a, label="P3")

    plt.xlabel("time")
    plt.ylabel("Activity")
    plt.legend()
    plt.show()

def plot_result(population, pop_id):
    if pop_id in [1, 2]:
        y_size = exc_y_size
    else:
        y_size = inh_y_size

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
            data_source['group'].append("n")

    print(population.name, spike_count / (dt * simulation_steps))

    df = pd.DataFrame(data=data_source)

    sns.scatterplot(x="time", y="neuron_idx", data=df, hue="group", s=5)
    plt.title("Raster Plot")
    plt.ylabel("Neuron")
    plt.xlabel("time")
    plt.show()

    def sort_func(x):
        return x[0]

    if pop_id == 1:
        c_samples = c_samples_exc_1
    elif pop_id == 2:
        c_samples = c_samples_exc_2
    else:
        return

    c_items = list(c_samples.items())
    c_items.sort(key=sort_func)
    ts = [x[0] for x in c_items]
    cs = [x[1] for x in c_items]
    plt.plot(ts, cs)
    plt.xlabel('time')
    plt.ylabel('C(t)')
    plt.show()

def connect_neurons(e1, e2, i1):
    connect_neurons_t1(e1, e2, i1)
    pass

def connect_neurons_t1(e1, e2, i1):
    c1 = RandomConnectorFixedProb(pre=e1, post=e1, w_mu=2, w_sigma=0.1, w_mu_inh=0, w_sigma_inh=0, d=1, is_exc_func=lambda x: True, prob=0.02)
    c1.connect(context)

    c2 = RandomConnectorFixedProb(pre=e2, post=e2, w_mu=2, w_sigma=0.1, w_mu_inh=0, w_sigma_inh=0, d=1, is_exc_func=lambda x: True, prob=0.02)
    c2.connect(context)

    c3 = RandomConnectorFixedProb(pre=e1, post=i1, w_mu=2, w_sigma=0.1, w_mu_inh=0, w_sigma_inh=0, d=1, is_exc_func=lambda x: True, prob=0.02)
    c3.connect(context)
    c4 = RandomConnectorFixedProb(pre=e2, post=i1, w_mu=2, w_sigma=0.1, w_mu_inh=0, w_sigma_inh=0, d=1, is_exc_func=lambda x: True, prob=0.02)
    c4.connect(context)

    # c5 = RandomConnectorFixedProb(pre=e1, post=e2, w_mu=0.5, w_sigma=0.1, w_mu_inh=0, w_sigma_inh=0, d=1, is_exc_func=lambda x: True, prob=0.01)
    # c5.connect(context)

    # c6 = RandomConnectorFixedProb(pre=e2, post=e1, w_mu=0.5, w_sigma=0.1, w_mu_inh=0, w_sigma_inh=0, d=1, is_exc_func=lambda x: True, prob=0.01)
    # c6.connect(context)

    c7 = RandomConnectorFixedProb(pre=i1, post=e1, w_mu=0, w_sigma=0, w_mu_inh=-1, w_sigma_inh=0.1, d=1, is_exc_func=lambda x: False, prob=0.01)
    c7.connect(context)
    c8 = RandomConnectorFixedProb(pre=i1, post=e2, w_mu=0, w_sigma=0, w_mu_inh=-1, w_sigma_inh=0.1, d=1, is_exc_func=lambda x: False, prob=0.01)
    c8.connect(context)

def main():
    global context

    context = Context(dt=dt)

    e1 = Population("e1", (exc_x_size, exc_y_size), context, get_neuron_init(context))
    e1.set_pop_in_c(get_pop_in_c(1))

    e2 = Population("e2", (exc_x_size, exc_y_size), context, get_neuron_init(context))
    e2.set_pop_in_c(get_pop_in_c(2))

    i1 = Population("i1", (inh_x_size, inh_y_size), context, get_neuron_init(context, is_exc=False))

    connect_neurons(e1, e2, i1)

    print("simulating...")

    for i in range(simulation_steps):
        e1.steps(1)
        e2.steps(1)
        i1.steps(1)
        context.step()
        print(i)

    # plot_result(e1, 1)
    # plot_result(e2, 2)
    # plot_result(i1, 3)
    plot_res(e1, e2, i1)


if __name__ == '__main__':
    main()

import numpy as np

from neurokit.context import Context
from neurokit.populations.population import Population
from neurokit.models.lif import LIF
from neurokit.learning_rule import RMSTDP, STDP

pattern_1 = [1, 1, 0, 0, 0, 0, 3, 4, 5, 0]
pattern_2 = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
last_applied_pattern = 2

# Simulation
steps = 10000

# Context
dt = 1
a_p = 0.1
a_n = -0.6
tau_p = 2
tau_n = 2
tau_c = 1
tau_d = 1

p_d = 100
n_d = -100
t_d = 2

# Neuron
tau = 1
u_r = -70
u_t = -50
r = 1

# Synapse
mu = 6
sigma = 0.05
d = 1

pop = None
context = None

show_spike = False


def spike_cb(name):
    if name == '0_10':
        if show_spike:
            print(name)

        if last_applied_pattern == 1:
            context.change_dopamine(n_d, t_d)
        else:
            context.change_dopamine(p_d, t_d)

    if name == '0_11':
        if show_spike:
            print(name)

        if last_applied_pattern == 1:
            context.change_dopamine(p_d, t_d)
        else:
            context.change_dopamine(n_d, t_d)


def get_neuron_init(context):
    def neuron_init(x, y):
        if y == 10:
            return LIF(context=context, tau=tau, u_r=u_r, u_t=-50, r=r, name=f"{x}_{y}", spike_cb=spike_cb)

        if y == 11:
            return LIF(context=context, tau=tau, u_r=u_r, u_t=-60, r=r, name=f"{x}_{y}", spike_cb=spike_cb)

        return LIF(context=context, tau=tau, u_r=u_r, u_t=u_t, r=r, name=f"{x}_{y}", spike_cb=spike_cb)

    return neuron_init


def print_weights(pop):
    d1 = pop.get_neuron(0, 10)
    d2 = pop.get_neuron(0, 11)
    for i in range(10):
        neu = pop.get_neuron(0, i)
        for s in neu._out_synapses:
            if s.dest is d1:
                print(i, 10, s.w)
            else:
                print(i, 11, s.w)
    print("")


def main():
    global last_applied_pattern, pop, context, show_spike

    learning_rule = RMSTDP(
        stdp_rule=STDP(a_p, a_n, tau_p, tau_n),
        tau_c=tau_c,
        tau_d=tau_d
    )
    # learning_rule = STDP(a_p, a_n, tau_p, tau_n)

    context = Context(dt=dt, learning_rule=learning_rule)

    pop = Population("pop", (1, 12), context, get_neuron_init(context))
    for i in range(10):
        pop.connect_two((0, i), (0, 10), w=np.random.normal(mu, sigma), d=d)
        pop.connect_two((0, i), (0, 11), w=np.random.normal(mu, sigma), d=d)

    print_weights(pop)

    for i in range(steps):
        if i % 100 == 0:
            pat = pattern_1
            ap = 1
            if last_applied_pattern == 1:
                pat = pattern_2
                ap = 2

            last_applied_pattern = ap

            for idx, j in enumerate(pat):
                n = pop.get_neuron(0, idx)
                if j != 0:
                    n.register_potential_change(40, context.t() + context.dt() * j)

        pop.steps(1)
        context.step()

    print_weights(pop)
    context._learning_rule = None

    for i in range(1000):
        pop.steps(1)
        context.step()

    show_spike = True

    print("\npat_1")
    for idx, j in enumerate(pattern_1):
        n = pop.get_neuron(0, idx)
        if j != 0:
            n.register_potential_change(40, context.t() + context.dt() * j)
    for i in range(20):
        pop.steps(1)
        context.step()
    print("end_pat_1")

    for i in range(1000):
        pop.steps(1)
        context.step()

    print("\npat_2")
    for idx, j in enumerate(pattern_2):
        n = pop.get_neuron(0, idx)
        if j != 0:
            n.register_potential_change(40, context.t() + context.dt() * j)
    for i in range(20):
        pop.steps(1)
        context.step()
    print("end_pat_2")


if __name__ == '__main__':
    main()

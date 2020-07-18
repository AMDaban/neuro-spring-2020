import numpy as np

from neurokit.context import Context
from neurokit.populations.population import Population
from neurokit.models.lif import LIF

pattern_1 = [1, 2, 2, 2, 1, 0, 0, 0, 0, 0]
pattern_2 = [0, 0, 0, 0, 0, 3, 1, 2, 1, 1]
last_applied_pattern = 2

# Simulation
steps = 10000

# Context
dt = 0.1
stdp_enabled = True
a_p = 0.1
a_n = -1
tau_p = 2
tau_n = 2

# Neuron
inp_tau = 5
inp_u_r = -70
inp_u_t = -50
inp_r = 1

out_tau = 3
out_u_r = -70
out_u_t = -60
out_r = 1

inh_tau = 2
inh_u_r = -70
inh_u_t = -50
inh_r = 1

# Synapse
mu = 3
sigma = 0.5
d = 1

inh_mu = 1
inh_sigma = 0

def get_neuron_init(context):
    def neuron_init(x, y):
        if y == 12:
            return LIF(context=context, tau=inh_tau, u_r=inh_u_r, u_t=inh_u_t, r=inh_r, name=f"({x}, {y})")
        elif y < 10:
            return LIF(context=context, tau=inp_tau, u_r=inp_u_r, u_t=inp_u_t, r=inp_r, name=f"({x}, {y})")
        else:
            return LIF(context=context, tau=out_tau, u_r=out_u_r, u_t=out_u_t, r=out_r, name=f"({x}, {y})")

    return neuron_init


def main():
    global last_applied_pattern

    context = Context(dt=dt, stdp_enabled=True, a_p=a_p, a_n=a_n, tau_p=tau_p, tau_n=tau_n)

    pop = Population("pop", (1, 13), context, get_neuron_init(context))
    for i in range(10):
        pop.connect_two((0, i), (0, 10), w=np.random.normal(mu, sigma), d=d+1)
        pop.connect_two((0, i), (0, 11), w=np.random.normal(mu, sigma), d=d+1)
        pop.connect_two((0, i), (0, 12), w=np.random.normal(mu, sigma), d=d)

    # pop.connect_two((0, 12), (0, 10), w=-1 * np.random.normal(inh_mu, inh_sigma), d=d)
    # pop.connect_two((0, 12), (0, 11), w=-1 * np.random.normal(inh_mu, inh_sigma), d=d)

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
                    n.register_potential_change(100, context.t() + context.dt() * j)

        pop.steps(1)
        context.step()

    # print(d2.get_monitor().get_observations())

    d1 = pop.get_neuron(0, 10)
    d2 = pop.get_neuron(0, 11)
    for i in range(10):
        neu = pop.get_neuron(0, i)
        for s in neu._out_synapses:
            if s.dest is d2:
                print(i, 10, s.w)
            else:
                print(i, 11, s.w)

    context.stdp_enabled = False

    for i in range(100):
        pop.steps(1)
        context.step()

    print("\npat_1")
    for idx, j in enumerate(pattern_1):
        n = pop.get_neuron(0, idx)
        if j != 0:
            n.register_potential_change(40, context.t() + context.dt() * j)
    for i in range(25):
        pop.steps(1)
        context.step()
    print("end_pat_1")

    for i in range(100):
        pop.steps(1)
        context.step()

    print("\npat_2")
    for idx, j in enumerate(pattern_2):
        n = pop.get_neuron(0, idx)
        if j != 0:
            n.register_potential_change(40, context.t() + context.dt() * j)
    for i in range(25):
        pop.steps(1)
        context.step()
    print("end_pat_2")


if __name__ == '__main__':
    main()

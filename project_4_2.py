import random
from collections import defaultdict

import numpy as np
from tqdm import trange

from neurokit.context import Context
from neurokit.populations.population import Population
from neurokit.models.lif import LIF
from neurokit.learning_rule import RMSTDP, STDP

# Neuron
tau = 1
u_r = -70
u_t = -50
r = 1

# Simulation
steps = 2500
size = (5, 200)
input_size = 50
pat_count = 2
pat_res_size = 20
pat_spike_prob = 0.8
pat_window_size = 20


def is_inhb(index):
    return index[0] >= (size[0] - 1)


# Context
dt = 1
a_p = 0.1
a_n = -0.6
tau_p = 2
tau_n = 2
tau_c = 1
tau_d = 1

# Dopamine
p_d = 5
n_d = -20
t_d = 1

# Connections
con_prob = 0.5
con_w_mu = 4
con_w_sigma = 0.1
con_w_mu_inhb = -30
con_w_sigma_ihb = 0.1
con_d_range = [1, 3]

context = None
pop = None
last_applied_pattern = -1
inp_neurons_global = []
pat_neurons_global = []
pats_global = []
mod = 0
window_spiked_neurons = defaultdict(int)
test_window_spikes = defaultdict(int)


# noinspection PyUnresolvedReferences
def test():
    global mod, test_window_spikes

    mod = 1

    for pat_idx, pat in enumerate(pats_global):
        test_window_spikes = defaultdict(int)

        for idx, j in enumerate(pat):
            neuron_idx = inp_neurons_global[idx]
            if j != 0:
                neuron = pop.get_neuron(neuron_idx[0], neuron_idx[1])
                neuron.register_potential_change(u_t - u_r + 10, context.t() + context.dt() * j)

        for i in range(pat_window_size + 5):
            pop.steps(1)
            context.step()

        max_spike_pat = -1
        max_spike = -1
        for idx, pat_neurons in enumerate(pat_neurons_global):
            sp_count = 0
            for neu in pat_neurons:
                if neu in test_window_spikes:
                    sp_count += test_window_spikes[neu]

            if sp_count > max_spike:
                max_spike = sp_count
                max_spike_pat = idx
        print(pat_idx, '->', max_spike_pat, 'with', max_spike)

        for i in range(pat_window_size + 5):
            pop.steps(1)
            context.step()

    mod = 0


# noinspection PyUnresolvedReferences
def spike_check():
    sum_spike = 0

    max_spike_pat = -1
    max_spike = -1
    for idx, pat_neurons in enumerate(pat_neurons_global):
        sp_count = 0
        for neu in pat_neurons:
            if neu in window_spiked_neurons:
                sp_count += window_spiked_neurons[neu]
                sum_spike += window_spiked_neurons[neu]

        if sp_count > max_spike:
            max_spike = sp_count
            max_spike_pat = idx

    if (max_spike_pat == last_applied_pattern) and ((max_spike / sum_spike) > 0.5):
        print(last_applied_pattern, max_spike_pat, max_spike, "rew", sum_spike)
        context.change_dopamine(p_d, t_d)
    else:
        print(last_applied_pattern, max_spike_pat, max_spike, "pun", sum_spike)
        context.change_dopamine(n_d, t_d)


# noinspection PyUnresolvedReferences
def simulate():
    global last_applied_pattern, window_spiked_neurons
    next_spike_check = -1

    for i in trange(steps):
        if i == next_spike_check:
            spike_check()

        if i % (pat_window_size + 5) == 0:
            window_spiked_neurons = defaultdict(int)
            last_applied_pattern = (last_applied_pattern + 1) % pat_count
            next_spike_check = i + pat_window_size + 1

            for idx, j in enumerate(pats_global[last_applied_pattern]):
                neuron_idx = inp_neurons_global[idx]
                if j != 0:
                    neuron = pop.get_neuron(neuron_idx[0], neuron_idx[1])
                    neuron.register_potential_change(u_t - u_r + 10, context.t() + context.dt() * j)

        pop.steps(1)
        context.step()


def spike_cb(name):
    splitted = name.split('_')

    if mod == 0:
        window_spiked_neurons[(int(splitted[0]), int(splitted[1]))] += 1
    else:
        test_window_spikes[(int(splitted[0]), int(splitted[1]))] += 1


def neuron_init(x, y):
    return LIF(context=context, tau=tau, u_r=u_r, u_t=u_t, r=r, name=f"{x}_{y}", spike_cb=spike_cb)


def random_index():
    _w, _h = size
    return random.randint(0, _w - 1), random.randint(0, _h - 1)


def connect_neurons(pop):
    def get_pair():
        _src = random_index()
        _dest = random_index()
        _description = f"{_src[0]}_{_src[1]}_{_dest[0]}_{_dest[1]}"
        return _src, _dest, _description

    total_neurons = size[0] * size[1]
    total_cons = int((total_neurons * (total_neurons - 1)) * con_prob)

    cons = set()
    for i in range(total_cons):
        src, dest, description = get_pair()
        while (src == dest) or (description in cons):
            src, dest, description = get_pair()

        cons.add(description)

        if is_inhb(src):
            w = np.random.normal(con_w_mu_inhb, con_w_sigma_ihb)
        else:
            w = np.random.normal(con_w_mu, con_w_sigma)
        d = random.randint(con_d_range[0], con_d_range[1])

        pop.connect_two(src, dest, w, d)


def choose_neurons():
    inp_neurons = []
    pat_neurons = []

    chosen_neurons = set()

    for i in range(input_size):
        idx = random_index()
        while (idx in chosen_neurons) or is_inhb(idx):
            idx = random_index()
        inp_neurons.append(idx)
        chosen_neurons.add(idx)

    for i in range(pat_count):
        pat_neurons.append([])
        for j in range(pat_res_size):
            idx = random_index()
            while (idx in chosen_neurons) or is_inhb(idx):
                idx = random_index()
            pat_neurons[i].append(idx)
            chosen_neurons.add(idx)

    return inp_neurons, pat_neurons


def make_patterns():
    pats = []

    for i in range(pat_count):
        pat = []
        for j in range(input_size):
            spike_time = 0
            if random.random() < pat_spike_prob:
                spike_time = random.randint(1, pat_window_size)
            pat.append(spike_time)

        pats.append(tuple(pat))

    return pats


def main():
    global inp_neurons_global, pat_neurons_global, pats_global, pop, context

    learning_rule = RMSTDP(
        stdp_rule=STDP(a_p, a_n, tau_p, tau_n),
        tau_c=tau_c,
        tau_d=tau_d
    )
    context = Context(dt=dt, learning_rule=learning_rule)

    print("creating  population ...")
    pop = Population("pop", size, context, neuron_init)

    print("creating connections ...")
    connect_neurons(pop)

    print("choosing neurons ...")
    inp_neurons_global, pat_neurons_global = choose_neurons()

    print("making patterns ...")
    pats_global = make_patterns()

    print("pre testing ...")
    test()

    print("simulating ...")
    simulate()

    print("post testing ...")
    test()


if __name__ == '__main__':
    main()

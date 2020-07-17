from neurokit.context import Context
from neurokit.populations.population import Population
from neurokit.models.lif import LIF

# Simulation
steps = 10000

# Context
dt = 0.001
stdp_enabled = True
a_p = 10
a_n = -10
tau_p = 5
tau_n = 5

# Neuron
tau = 8
u_r = -70
u_t = -50
r = 5
src_pop_in_c = 80
dest_pop_in_c = 22

# Synapse
w = 5
d = 1


def get_neuron_init(context):
    def neuron_init(x, y):
        return LIF(context=context, tau=tau, u_r=u_r, u_t=u_t, r=r)

    return neuron_init


def src_pop_in_c_func(t):
    return src_pop_in_c


def dest_pop_in_c_func(t):
    return dest_pop_in_c


def main():
    context = Context(dt=dt, stdp_enabled=stdp_enabled, a_p=a_p, a_n=a_n, tau_p=tau_p, tau_n=tau_n)

    pop = Population("pop", (1, 2), context, get_neuron_init(context))
    pop.connect_two((0, 0), (0, 1), w=w, d=d)

    src = pop.get_neuron(0, 0)
    src.name = 'src'
    src.set_in_c(src_pop_in_c_func)

    dest = pop.get_neuron(0, 1)
    dest.name = 'dest'
    dest.set_in_c(dest_pop_in_c_func)

    for i in range(steps):
        pop.steps(1)
        context.step()


if __name__ == '__main__':
    main()

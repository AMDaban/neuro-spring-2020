from neurokit.context import Context
from neurokit.populations.population import Population
from neurokit.models.lif import LIF


def get_neuron_init(context):
    def neuron_init(x, y):
        return LIF(context=context, tau=10, u_r=-70, u_t=-40, r=1)

    return neuron_init


def main():
    context = Context(dt=0.001, stdp_enabled=True, a_p=1, a_n=-1, tau_p=0.03, tau_n=0.05)

    pop = Population("pop", (1, 2), context, get_neuron_init(context))
    pop.connect_two((0, 0), (0, 1), w=10, d=1)

    src = pop.get_neuron(0, 0)
    dest = pop.get_neuron(0, 1)

    for i in range(10000):
        if (context.t() + context.dt()) != (context.t() + context.dt()):
            print("EEEEEEE")

        if i % 100 == 0:
            src.register_potential_change(10, context.t() + context.dt())

        if i % 3 == 0:
            dest.register_potential_change(10, context.t() + context.dt())

        pop.steps(1)
        context.step()

    spikes = pop.get_monitor().get_observations()


if __name__ == '__main__':
    main()

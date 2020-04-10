import numpy as np

from neurokit.models.lif import LIF
from matplotlib import pyplot as plt


def main():
    #  first practice
    first_case(tau=20, u_t=-10, u_r=-80, simulate_milliseconds=200, simulate_counts=6, practice=1,
               current_interval=2)
    first_case(tau=30, u_t=-10, u_r=-80, simulate_milliseconds=200, simulate_counts=6, practice=1,
               current_interval=3)
    first_case(tau=40, u_t=0, u_r=-80, simulate_milliseconds=200, simulate_counts=10, practice=1,
               current_interval=1)
    first_case(tau=50, u_t=-10, u_r=-80, simulate_milliseconds=200, simulate_counts=6, practice=1,
               current_interval=2)
    first_case(tau=60, u_t=-10, u_r=-80, simulate_milliseconds=200, simulate_counts=6, practice=1,
               current_interval=2)


def first_case(tau, u_t, u_r, simulate_milliseconds, simulate_counts, practice, current_interval):
    times = [t for t in range(0, simulate_milliseconds)]
    currents = []
    potentials = []
    f_values = []

    for c in np.arange(0, simulate_counts * current_interval, current_interval):
        def c_func(t):
            return c

        neuron = LIF(c_func, tau=tau, u_t=u_t, u_r=u_r)

        u_values, _, spike_times = neuron.simulate(simulate_milliseconds)

        potentials.append(u_values)
        currents.append([c for _ in range(0, simulate_milliseconds)])
        f_values.append(len(spike_times))

    plt.title("p{}, u_r={}, u_t={}, tau={}".format(practice, u_r, u_t, tau))
    plt.hlines(u_r, 0, simulate_milliseconds, linestyles='dashed', label='rest', colors='brown')
    plt.hlines(u_t, 0, simulate_milliseconds, linestyles='dashed', label='threshold', colors='red')
    for i in range(0, simulate_counts):
        plt.plot(times, potentials[i])
    plt.xlabel('time')
    plt.ylabel('U(t)')
    plt.show()
    plt.close()

    plt.title("p{}, u_r={}, u_t={}, tau={}".format(practice, u_r, u_t, tau))
    for i in range(0, simulate_counts):
        plt.plot(times, currents[i])
    plt.xlabel('time')
    plt.ylabel('I(t)')
    plt.show()
    plt.close()

    plt.title("p{}, u_r={}, u_t={}, tau={}".format(practice, u_r, u_t, tau))
    plt.plot([c for c in np.arange(0, simulate_counts * current_interval, current_interval)], f_values)
    plt.xlabel('I')
    plt.ylabel('F')
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()

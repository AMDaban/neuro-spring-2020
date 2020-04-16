import random
import math

import numpy as np

from neurokit.models.lif import LIF
from neurokit.models.exponential_lif import ExponentialLIF
from neurokit.models.adaptive_exponential_lif import AdaptiveExponentialLIF
from neurokit.models.lif import LIF
from matplotlib import pyplot as plt

random_c_func_maximum_c = 10
random_c_func_maximum_c_change = 0.1
first_non_zero_c_time = 5
last_non_zero_c_time = 70
steps = 100000
dt = 0.001
first_test_c = 5
second_test_c = 10
f_c_range = np.arange(0, 50, 1)


def get_c_func(c):
    def c_func(t):
        if t < first_non_zero_c_time or t > last_non_zero_c_time:
            return 0
        return c

    return c_func


def get_random_c_func():
    c = 0

    def c_func(t):
        nonlocal c

        if t < first_non_zero_c_time or t > last_non_zero_c_time:
            c = 0
        else:
            c += (random.random() - 0.5) * random_c_func_maximum_c_change
            if c < 0:
                c = 0
            if c > random_c_func_maximum_c:
                c = random_c_func_maximum_c

        return c

    return c_func


def plot_u_t(u_values, t_values, u_r, u_t):
    plt.hlines(u_r, 0, steps * dt, linestyles='dashed', label='rest', colors='brown')
    plt.hlines(u_t, 0, steps * dt, linestyles='dashed', label='threshold', colors='red')
    for i in range(len(u_values)):
        plt.plot(t_values, u_values[i])
    plt.xlabel('time')
    plt.ylabel('U(t)')
    plt.show()
    plt.close()


def plot_c_t(c_values, t_values):
    for i in range(len(c_values)):
        plt.plot(t_values, c_values[i])
    plt.xlabel('time')
    plt.ylabel('C(t)')
    plt.show()
    plt.close()


def plot_f_c(f_values, c_values):
    plt.plot(c_values, f_values)
    plt.xlabel('C(t)')
    plt.ylabel('F(t)')
    plt.show()
    plt.close()


def test_p1(tau, u_r, u_t, r):
    all_u_values = []
    all_c_values = []

    c_func = get_c_func(first_test_c)
    neuron = LIF(c_func, tau=tau, u_r=u_r, u_t=u_t, r=r, dt=dt)
    neuron.steps(steps)
    times, u_values, c_values, _ = neuron.get_monitor().get_observations()
    all_u_values.append(u_values)
    all_c_values.append(c_values)

    c_func = get_c_func(second_test_c)
    neuron = LIF(c_func, tau=tau, u_r=u_r, u_t=u_t, r=r, dt=dt)
    neuron.steps(steps)
    times, u_values, c_values, _ = neuron.get_monitor().get_observations()
    all_u_values.append(u_values)
    all_c_values.append(c_values)

    plot_c_t(all_c_values, times)
    plot_u_t(all_u_values, times, u_r, u_t)

    c_values = []
    f_values = []

    for c in f_c_range:
        c_func = get_c_func(c)
        neuron = LIF(c_func, tau=tau, u_r=u_r, u_t=u_t, dt=dt)
        neuron.steps(steps)
        _, _, _, spikes = neuron.get_monitor().get_observations()
        c_values.append(c)
        f_values.append(len(spikes))

    plot_f_c(f_values, c_values)


def test_p2(tau, u_r, u_t, r):
    all_u_values = []
    all_c_values = []

    c_func = get_random_c_func()
    neuron = LIF(c_func, tau=tau, u_r=u_r, u_t=u_t, r=r, dt=dt)
    neuron.steps(steps)
    times, u_values, c_values, _ = neuron.get_monitor().get_observations()
    all_u_values.append(u_values)
    all_c_values.append(c_values)

    plot_c_t(all_c_values, times)
    plot_u_t(all_u_values, times, u_r, u_t)


def test_p3_part1(tau, u_r, u_t, r, delta_t, theta_rh):
    all_u_values = []
    all_c_values = []

    c_func = get_c_func(first_test_c)
    neuron = ExponentialLIF(c_func, tau=tau, u_r=u_r, u_t=u_t, r=r, dt=dt, delta_t=delta_t, theta_rh=theta_rh)
    neuron.steps(steps)
    times, u_values, c_values, _ = neuron.get_monitor().get_observations()
    all_u_values.append(u_values)
    all_c_values.append(c_values)

    c_func = get_c_func(second_test_c)
    neuron = ExponentialLIF(c_func, tau=tau, u_r=u_r, u_t=u_t, r=r, dt=dt, delta_t=delta_t, theta_rh=theta_rh)
    neuron.steps(steps)
    times, u_values, c_values, _ = neuron.get_monitor().get_observations()
    all_u_values.append(u_values)
    all_c_values.append(c_values)

    plot_c_t(all_c_values, times)
    plot_u_t(all_u_values, times, u_r, u_t)

    c_values = []
    f_values = []

    for c in f_c_range:
        c_func = get_c_func(c)
        neuron = ExponentialLIF(c_func, tau=tau, u_r=u_r, u_t=u_t, dt=dt, delta_t=delta_t, theta_rh=theta_rh)
        neuron.steps(steps)
        _, _, _, spikes = neuron.get_monitor().get_observations()
        c_values.append(c)
        f_values.append(len(spikes))

    plot_f_c(f_values, c_values)


def test_p3_part2(tau, u_r, u_t, r, delta_t, theta_rh):
    all_u_values = []
    all_c_values = []

    c_func = get_random_c_func()
    neuron = ExponentialLIF(c_func, tau=tau, u_r=u_r, u_t=u_t, r=r, dt=dt, delta_t=delta_t, theta_rh=theta_rh)
    neuron.steps(steps)
    times, u_values, c_values, _ = neuron.get_monitor().get_observations()
    all_u_values.append(u_values)
    all_c_values.append(c_values)

    plot_c_t(all_c_values, times)
    plot_u_t(all_u_values, times, u_r, u_t)


def test_p4_part1(tau_m, tau_w, u_r, u_t, r, delta_t, theta_rh, a, b):
    all_u_values = []
    all_c_values = []

    c_func = get_c_func(first_test_c)
    neuron = AdaptiveExponentialLIF(c_func, tau_m=tau_m, tau_w=tau_w, u_r=u_r, u_t=u_t, r=r, dt=dt, delta_t=delta_t,
                                    theta_rh=theta_rh, a=a, b=b)
    neuron.steps(steps)
    times, u_values, c_values, _ = neuron.get_monitor().get_observations()
    all_u_values.append(u_values)
    all_c_values.append(c_values)

    c_func = get_c_func(second_test_c)
    neuron = AdaptiveExponentialLIF(c_func, tau_m=tau_m, tau_w=tau_w, u_r=u_r, u_t=u_t, r=r, dt=dt, delta_t=delta_t,
                                    theta_rh=theta_rh, a=a, b=b)
    neuron.steps(steps)
    times, u_values, c_values, _ = neuron.get_monitor().get_observations()
    all_u_values.append(u_values)
    all_c_values.append(c_values)

    plot_c_t(all_c_values, times)
    plot_u_t(all_u_values, times, u_r, u_t)

    c_values = []
    f_values = []

    for c in f_c_range:
        c_func = get_c_func(c)
        neuron = AdaptiveExponentialLIF(c_func, tau_m=tau_m, tau_w=tau_w, u_r=u_r, u_t=u_t, r=r, dt=dt, delta_t=delta_t,
                                        theta_rh=theta_rh, a=a, b=b)
        neuron.steps(steps)
        _, _, _, spikes = neuron.get_monitor().get_observations()
        c_values.append(c)
        f_values.append(len(spikes))

    plot_f_c(f_values, c_values)


def test_p4_part2(tau_m, tau_w, u_r, u_t, r, delta_t, theta_rh, a, b):
    all_u_values = []
    all_c_values = []

    c_func = get_random_c_func()
    neuron = AdaptiveExponentialLIF(c_func, tau_m=tau_m, tau_w=tau_w, u_r=u_r, u_t=u_t, r=r, dt=dt, delta_t=delta_t,
                                    theta_rh=theta_rh, a=a, b=b)
    neuron.steps(steps)
    times, u_values, c_values, _ = neuron.get_monitor().get_observations()
    all_u_values.append(u_values)
    all_c_values.append(c_values)

    plot_c_t(all_c_values, times)
    plot_u_t(all_u_values, times, u_r, u_t)


def main():
    # practice 1
    test_p1(tau=10, u_r=-80, u_t=0, r=10)
    test_p1(tau=20, u_r=-80, u_t=0, r=10)
    test_p1(tau=30, u_r=-80, u_t=0, r=10)
    test_p1(tau=40, u_r=-80, u_t=0, r=10)
    test_p1(tau=50, u_r=-80, u_t=0, r=10)

    # practice 2
    test_p2(tau=10, u_r=-80, u_t=0, r=10)
    test_p2(tau=10, u_r=-80, u_t=-10, r=10)
    test_p2(tau=10, u_r=-80, u_t=-15, r=10)
    test_p2(tau=10, u_r=-80, u_t=0, r=15)
    test_p2(tau=20, u_r=-80, u_t=0, r=10)

    # practice 3  - part 1
    test_p3_part1(tau=10, u_r=-80, u_t=0, r=10, delta_t=1, theta_rh=1)
    test_p3_part1(tau=20, u_r=-80, u_t=0, r=10, delta_t=1, theta_rh=1)
    test_p3_part1(tau=30, u_r=-80, u_t=0, r=10, delta_t=1, theta_rh=1)
    test_p3_part1(tau=40, u_r=-80, u_t=0, r=10, delta_t=1, theta_rh=1)
    test_p3_part1(tau=50, u_r=-80, u_t=0, r=10, delta_t=1, theta_rh=1)

    # practice 3 - part 2
    test_p3_part2(tau=10, u_r=-80, u_t=0, r=10, delta_t=1, theta_rh=1)
    test_p3_part2(tau=10, u_r=-80, u_t=-10, r=10, delta_t=1, theta_rh=1)
    test_p3_part2(tau=10, u_r=-80, u_t=-20, r=10, delta_t=1, theta_rh=1)
    test_p3_part2(tau=10, u_r=-80, u_t=-15, r=10, delta_t=1, theta_rh=1)
    test_p3_part2(tau=20, u_r=-80, u_t=0, r=10, delta_t=1, theta_rh=1)

    # practice 4  - part 1
    test_p4_part1(tau_m=10, tau_w=10, u_r=-80, u_t=0, r=10, delta_t=1, theta_rh=1, a=1, b=1)
    test_p4_part1(tau_m=20, tau_w=20, u_r=-80, u_t=0, r=10, delta_t=1, theta_rh=1, a=1, b=1)
    test_p4_part1(tau_m=30, tau_w=30, u_r=-80, u_t=0, r=10, delta_t=1, theta_rh=1, a=1, b=1)
    test_p4_part1(tau_m=40, tau_w=40, u_r=-80, u_t=0, r=10, delta_t=1, theta_rh=1, a=1, b=1)
    test_p4_part1(tau_m=50, tau_w=50, u_r=-80, u_t=0, r=10, delta_t=1, theta_rh=1, a=1, b=1)

    # practice 4 - part 2
    test_p4_part2(tau_m=10, tau_w=10, u_r=-80, u_t=0, r=10, delta_t=1, theta_rh=1, a=1, b=1)
    test_p4_part2(tau_m=10, tau_w=10, u_r=-80, u_t=-10, r=10, delta_t=1, theta_rh=1, a=1, b=1)
    test_p4_part2(tau_m=10, tau_w=10, u_r=-80, u_t=-20, r=10, delta_t=1, theta_rh=1, a=1, b=1)
    test_p4_part2(tau_m=10, tau_w=10, u_r=-80, u_t=-15, r=10, delta_t=1, theta_rh=1, a=1, b=1)
    test_p4_part2(tau_m=20, tau_w=20, u_r=-80, u_t=0, r=10, delta_t=1, theta_rh=1, a=1, b=1)


if __name__ == '__main__':
    main()

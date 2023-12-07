import argparse
from itertools import repeat

import numpy as np

from IsingEvolution import IsingEvol
from multiprocessing import Pool
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pickle


def evolve_system(N, dt, h, J, trotter_steps, obs_Z, obs_XX, gpu=False, linear_increase=True, samples=1, inverse=False):
    # Initialization
    ising_model_evolution = IsingEvol(N, dt, h, J, gpu=gpu, inverse=inverse)
    ising_model_evolution.observables(obs_Z, obs_XX)

    # Execution
    states = ising_model_evolution.execute(draw=False, steps=trotter_steps, linear_increase=linear_increase,
                                           samples=samples)
    expectations = ising_model_evolution.compute_expectationvals(states)
    ising_model_evolution.plot(expectations)


def iteration_kinks(ising_model_evolution, steps, samples, h=None):
    states = ising_model_evolution.execute(draw=False, steps=steps, samples=samples, h=h)
    nok_mean, nok_sig = ising_model_evolution.compute_kink_density(states)
    return [nok_mean, nok_sig]


def estimate_kinks_tau_dependency(N, dt, h, J, trotter_steps, gpu=False, samples=1, periodic=False, inverse=False):
    ising_model_evolution = IsingEvol(N, dt, h, J, gpu=gpu, periodic=periodic, inverse=inverse)
    ising_model_evolution.progress = False
    steps = range(1, trotter_steps + 1)
    tau = dt * np.array(steps)

    with Pool() as pool:
        nok = np.array(
            pool.starmap(iteration_kinks, zip(repeat(ising_model_evolution), steps, repeat(samples), repeat(None))))

    kdens_mean = nok[:, 0]
    kdens_sig = nok[:, 1]

    filename = f'kinks_N{N}_J{J}_h{h}_dt{dt}_steps{trotter_steps}_periodic{periodic}'
    plot_dependency(filename, np.array(tau), np.array(kdens_mean), np.array(kdens_sig))
    things_to_save = [tau, kdens_mean, kdens_sig]

    with open("../data/" + filename, "wb") as f:
        pickle.dump(things_to_save, f)


def estimate_kinks_magnetic_dependency(N, dt, h, J, trotter_steps, gpu=False, samples=1, periodic=False, inverse=False):
    h_array = np.linspace(0, h, trotter_steps)
    ising_model_evolution = IsingEvol(N, dt, h, J, gpu=gpu, periodic=periodic, inverse=inverse)
    ising_model_evolution.progress = False

    iteration_kinks(ising_model_evolution, 100, 1, 0)

    with Pool() as pool:
        nok = np.array(pool.starmap(iteration_kinks,
                                    zip(repeat(ising_model_evolution), repeat(trotter_steps), repeat(samples),
                                        h_array)))

    # nok = ising_model_evolution.compute_kink_density(
    #     ising_model_evolution.execute(draw=False, steps=trotter_steps, samples=samples, h=h), raw=True)

    kdens_mean = nok[:, 0]
    kdens_sig = nok[:, 1]

    filename = f'kinks_N{N}_J{J}_h_max{h}_dt{dt}_steps{trotter_steps}_periodic{periodic}'
    plot_dependency(filename, np.array(h_array), np.array(kdens_mean), np.array(kdens_sig), xlabel="Magnetic field h")
    # plot_dependency(filename, np.array(h_array), np.array(nok), np.array(trotter_steps * [0]),
    #                 xlabel="Magnetic field h")
    things_to_save = [h_array, np.array(nok), np.array(trotter_steps * [0])]

    with open("../data/" + filename, "wb") as f:
        pickle.dump(things_to_save, f)


def plot_dependency(filename, tau, kdens_mean, kdens_sig, plot_fit=False, a=0, e=0, xlabel=None):
    plt.plot(tau, kdens_mean)
    print(e)
    if plot_fit:
        plt.plot(tau, kink_density_theory(tau, a, e),
                 label=f"$f(\\tau)={round(a, 2)} \\cdot \\tau ** ({round(e, 2)})$")

        plt.plot(tau, kink_density_theory(tau, 1 / (2 * np.pi * np.sqrt(2)), -0.5), label="Theory", linestyle="--")

    plt.fill_between(tau, kdens_mean - kdens_sig, kdens_mean + kdens_sig, alpha=0.2)
    plt.xlabel(xlabel or '$\\tau_Q$')
    plt.ylabel('Kink density')
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../figs/" + filename + ".png")


def kink_density_theory(tau, a, e):
    return a * tau ** e


def fit_kinks(tau, kinks, kinks_sigma):
    return curve_fit(kink_density_theory, tau, kinks, sigma=kinks_sigma, maxfev=5000)


def load_file(filename):
    with open(filename, "rb") as f:
        things_to_load = pickle.load(f)

    [tau, kdens_mean, kdens_sig] = things_to_load
    return tau, kdens_mean, kdens_sig


if __name__ == '__main__':
    parser = argparse.ArgumentParser("QuantumQuench")
    parser.add_argument("N", help="Number of Qubits", type=int)
    parser.add_argument("trotter_steps", help="The number of trotter steps", type=int)

    parser.add_argument("-dt", help="Duration of a trotter imestep", default=0.1, type=float)
    parser.add_argument("-Jv", "--J_values", help="Coupling strength J ", type=float, default=-0.25)
    parser.add_argument("-hv", "--h_values",
                        help="Coupling strength with external magnetic field. If linear increase is true, this will be the maximum h value",
                        type=float, default=-0.2)
    parser.add_argument("-li", "--linear_increase", help="If magnetic field increases linearly", type=bool,
                        default=True)

    parser.add_argument("-m", "--mode",
                        help="0 for kink density estimation, 1 for plain evolution of a given system and 2 for kink density magnetic field estimation",
                        type=int,
                        default=0)
    parser.add_argument("-gpu", "--gpu", help="1 to use GPU, else CPU", type=int, default=0)
    parser.add_argument("-s", "--samples", help="Number of samples to get from a run", type=int,
                        default=1)
    parser.add_argument("-p", "--periodic", help="1 for Periodic boundary conditions else Open", type=int,
                        default=0)
    parser.add_argument("-inv", "--inverse", help="1 for inverse magnetic field ramp h_0 -> 0 else 0 -> h_0", type=int,
                        default=0)

    args = parser.parse_args()

    N_ = args.N
    trotter_steps_ = args.trotter_steps
    J_ = args.J_values
    h_ = args.h_values
    dt_ = args.dt  # time step
    obs_Z_ = [5, 6]
    obs_XX_ = [[4, 14], [5, 6]]
    data_start = 10
    gpu_usage = bool(args.gpu)
    samples_ = args.samples
    mode = args.mode
    periodic = bool(args.periodic)
    inverse = bool(args.inverse)

    if mode == 0:
        estimate_kinks_tau_dependency(N_, dt_, h_, J_, trotter_steps_, gpu=bool(args.gpu), samples=samples_,
                                      periodic=periodic, inverse=inverse)

        filename = f"kinks_N{N_}_J{J_}_h{h_}_dt{dt_}_steps{trotter_steps_}_periodic{periodic}"
        tau, kinks_mean, kinks_sig = np.array(load_file("../data/" + filename))
        popt, pcov = fit_kinks(tau[data_start:], kinks_mean[data_start:], kinks_sig[data_start:] + 1e-15)
        plot_dependency("../figs/" + filename, tau, kinks_mean, kinks_sig, plot_fit=True, a=popt[0], e=popt[1])
    elif mode == 1:
        evolve_system(N_, dt_, h_, J_, trotter_steps_, obs_Z_, obs_XX_, gpu=bool(args.gpu), samples=samples_,
                      periodic=periodic, inverse=inverse)
    elif mode == 2:
        estimate_kinks_magnetic_dependency(N_, dt_, h_, J_, trotter_steps_, gpu=bool(args.gpu), samples=samples_,
                                           periodic=periodic, inverse=inverse)

    # plot_dependency("../figs/kinks_N20_J-0.25_h-1.5_dt0.1_steps100", tau, kinks_mean, kinks_sig)

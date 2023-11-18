import argparse
from itertools import repeat

import numpy as np

from IsingEvolution import IsingEvol
from multiprocessing import Pool
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pickle


def evolve_system(N, dt, h, J, trotter_steps, obs_Z, obs_XX, linear_increase=True):
    # Initialization
    ising_model_evolution = IsingEvol(N, dt, h, J)
    ising_model_evolution.observables(obs_Z, obs_XX)

    # Execution
    states = ising_model_evolution.execute(draw=False, steps=trotter_steps, linear_increase=linear_increase)
    expectations = ising_model_evolution.compute_expectationvals(states)
    ising_model_evolution.plot(expectations)


def iteration_kinks(ising_model_evolution, steps):
    states = ising_model_evolution.execute(draw=False, steps=steps)
    nok_mean, nok_sig = ising_model_evolution.compute_kink_density(states)
    return [nok_mean, nok_sig]


def estimate_kinks_tau_dependency(N, dt, h, J, trotter_steps):
    ising_model_evolution = IsingEvol(N, dt, h, J)
    ising_model_evolution.progress = False
    steps = range(1, trotter_steps + 1)
    tau = dt * np.array(steps)

    with Pool() as pool:
        nok = np.array(list(pool.starmap(iteration_kinks, zip(repeat(ising_model_evolution), steps))))

    kdens_mean = nok[:, 0]
    kdens_sig = nok[:, 1]

    filename = f'kinks_N{N}_J{J}_h{h}_dt{dt}_steps{trotter_steps}'
    plot_dependency(filename, np.array(tau), np.array(kdens_mean), np.array(kdens_sig))
    things_to_save = [tau, kdens_mean, kdens_sig]

    with open("../data/" + filename, "wb") as f:
        pickle.dump(things_to_save, f)


def plot_dependency(filename, tau, kdens_mean, kdens_sig, plot_fit=False, a=0, e=0, c=0):
    plt.plot(tau, kdens_mean)
    if plot_fit:
        plt.plot(tau, kink_density_theory(tau, a, e, c), label=f"a={round(a, 2)}, e={round(e, 2)}, c={round(c, 2)}")

    plt.fill_between(tau, kdens_mean - kdens_sig, kdens_mean + kdens_sig, alpha=0.2)
    plt.xlabel('$\\tau_Q$')
    plt.ylabel('Kink density')
    # plt.yscale("log")
    # plt.xscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../figs/" + filename + ".png")


def kink_density_theory(tau, a, e, c):
    return a * tau ** e + c


def fit_kinks(tau, kinks, kinks_sigma):
    return curve_fit(kink_density_theory, tau, kinks, sigma=kinks_sigma)


def load_file(filename):
    with open(filename, "rb") as f:
        things_to_load = pickle.load(f)

    [tau, kdens_mean, kdens_sig] = things_to_load
    return tau, kdens_mean, kdens_sig


if __name__ == '__main__':
    parser = argparse.ArgumentParser("QuantumQuench")
    parser.add_argument("N", help="Number of Qubits", type=int)
    parser.add_argument("trotter_steps", help="The number of trotter steps", type=int)

    parser.add_argument("-dt", help="Duration of a trotter imestep", default=0.1, type=int)
    parser.add_argument("-Jv", "--J_values", help="Coupling strength J ", type=int, default=-0.25)
    parser.add_argument("-hv", "--h_values",
                        help="Coupling strength with external magnetic field. If linear increase is true, this will be the maximum h value",
                        type=int, default=-0.2)
    parser.add_argument("-li", "--linear_increase", help="If magnetic field increases linearly", type=bool,
                        default=True)

    parser.add_argument("-m", "--mode",
                        help="0 for kink density estimation and 1 for plain evolution of a given system", type=int,
                        default=0)

    args = parser.parse_args()

    N_ = args.N
    trotter_steps_ = args.trotter_steps
    J_ = args.J_values
    h_ = args.h_values
    dt_ = args.dt  # time step
    obs_Z_ = [5, 6]
    obs_XX_ = [[4, 14], [5, 6]]
    data_start = 0

    if not args.mode:
        estimate_kinks_tau_dependency(N_, dt_, h_, J_, trotter_steps_)
    else:
        evolve_system(N_, dt_, h_, J_, trotter_steps_, obs_Z_, obs_XX_)

    # plot_dependency("../figs/kinks_N20_J-0.25_h-1.5_dt0.1_steps100", tau, kinks_mean, kinks_sig)

    # filename = "kinks_N20_J-0.25_h-0.2_dt0.1_steps100"
    # tau, kinks_mean, kinks_sig = np.array(load_file("../data/" + filename))
    # popt, pcov = fit_kinks(tau[data_start:], kinks_mean[data_start:], kinks_sig[data_start:] + 1e-15)
    # plot_dependency("../figs/" + filename, tau, kinks_mean, kinks_sig, plot_fit=True, a=popt[0],
    #                 e=popt[1], c=popt[2])

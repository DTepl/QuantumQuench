import argparse
from itertools import repeat

import numpy as np

from IsingEvolution import IsingEvol
from multiprocessing import Pool
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pickle


def evolve_system(N, dt, h, J, trotter_steps, obs_Z, obs_XX, gpu=False, periodic=False, samples=1,
                  inverse=False, bias=None):
    # Initialization
    ising_model_evolution = IsingEvol(N, dt, h, J, gpu=gpu, inverse=inverse, bias=bias, periodic=periodic)
    ising_model_evolution.observables(obs_Z, obs_XX)

    # Execution
    states = ising_model_evolution.execute(draw=False, steps=trotter_steps, samples=samples)
    expectations = ising_model_evolution.compute_expectationvals(states)
    ising_model_evolution.plot(expectations)


def iteration_kinks(ising_model_evolution, steps, samples, h=None):
    states = ising_model_evolution.execute(draw=False, steps=steps, samples=samples, h=h)
    nok_mean, nok_sig = ising_model_evolution.compute_kink_density(states)
    return [nok_mean, nok_sig]


def iteration_fidelity_simulation(N, h, J, dt, steps, samples, periodic, inverse, bias, gpu):
    ising_model_evolution = IsingEvol(N, dt, h, J, gpu=gpu, periodic=periodic, inverse=inverse, bias=bias)
    ising_model_evolution.progress = False
    states = ising_model_evolution.execute(draw=False, steps=steps, samples=samples)

    arr = []
    for step in range(1, len(states) + 1):
        arr.append(states[str(step)].data)

    return np.array(arr)


def iteration_fidelity_groundstate_dot(ising_obj, state, step, steps):
    groundstate = ising_obj.groundState(step, steps)
    return (np.abs(np.conj(state) @ groundstate) ** 2).flatten()


def estimate_kinks_tau_dependency(N, dt, h, J, trotter_steps, gpu=False, samples=1, periodic=False, inverse=False,
                                  bias=None):
    ising_model_evolution = IsingEvol(N, dt, h, J, gpu=gpu, periodic=periodic, inverse=inverse, bias=bias)
    ising_model_evolution.progress = False
    steps = range(1, trotter_steps + 1)
    tau = dt * np.array(steps)

    with Pool() as pool:
        nok = np.array(
            pool.starmap(iteration_kinks, zip(repeat(ising_model_evolution), steps, repeat(samples), repeat(None))))

    kdens_mean = nok[:, 0]
    kdens_sig = nok[:, 1]

    filename = f'kinks_N{N}_J{J}_h{h}_dt{dt}_steps{trotter_steps}_periodic{periodic}_inv{inverse}_bias{bias}'
    things_to_save = [tau, kdens_mean, kdens_sig]

    with open("../data/" + filename, "wb") as f:
        pickle.dump(things_to_save, f)


def estimate_kinks_magnetic_dependency(N, dt, h, J, trotter_steps, gpu=False, samples=1, periodic=False, inverse=False,
                                       bias=None):
    h_array = np.linspace(0, h, 100)
    ising_model_evolution = IsingEvol(N, dt, h, J, gpu=gpu, periodic=periodic, inverse=inverse, bias=bias)
    ising_model_evolution.progress = False

    iteration_kinks(ising_model_evolution, trotter_steps, 1, 0)

    with Pool() as pool:
        nok = np.array(pool.starmap(iteration_kinks,
                                    zip(repeat(ising_model_evolution), repeat(trotter_steps), repeat(samples),
                                        h_array)))

    # nok = ising_model_evolution.compute_kink_density(
    #     ising_model_evolution.execute(draw=False, steps=trotter_steps, samples=samples, h=h), raw=True)

    kdens_mean = nok[:, 0]
    kdens_sig = nok[:, 1]

    filename = f'kinks_N{N}_J{J}_h_max{h}_dt{dt}_steps{trotter_steps}_periodic{periodic}_inv{inverse}_bias{bias}'
    plot_dependency(filename, np.array(h_array), np.array(kdens_mean), np.array(kdens_sig), xlabel="Magnetic field h")
    # plot_dependency(filename, np.array(h_array), np.array(nok), np.array(trotter_steps * [0]),
    #                 xlabel="Magnetic field h")
    things_to_save = [h_array, np.array(nok), np.array(trotter_steps * [0])]

    with open("../data/" + filename, "wb") as f:
        pickle.dump(things_to_save, f)


def fidelity_measurement(N, h, J, steps, dt, gpu=False, periodic=False, inverse=False, bias=None):
    params = zip(repeat(N), repeat(h), repeat(J), dt, repeat(steps), repeat(steps), repeat(periodic), repeat(inverse),
                 repeat(bias), repeat(gpu))

    with Pool() as pool:
        states = np.array(pool.starmap(iteration_fidelity_simulation, params))

        groundstate_computer = IsingEvol(N, 0, h, J, gpu=gpu, periodic=periodic, inverse=inverse, bias=bias)
        # Swap axes from (5,100,-1) to (100,5,-1) where 5 are the number of dt's and 100 are the steps
        fidelities = pool.starmap(iteration_fidelity_groundstate_dot,
                                  zip(repeat(groundstate_computer), np.swapaxes(states, 0, 1), range(1, steps + 1),
                                      repeat(steps)))

    filename = f'fidelity_N{N}_J{J}_h{h}_steps{steps}_periodic{periodic}_inv{inverse}_bias{bias}'
    things_to_save = [steps, np.swapaxes(fidelities, 0, 1), 0]  # swap back for easier plotting

    with open("../data/" + filename, "wb") as f:
        pickle.dump(things_to_save, f)


def plot_dependency(filename, tau, kdens_mean, kdens_sig, plot_fit=False, a=0, e=0, g=0, xlabel=None, data_start=0,
                    data_end=-1):
    plt.plot(tau, kdens_mean)
    print(f"exponent: {e}")
    print(f"parallel magnetic: {(g / 6.89) ** (15 / 16)}")
    if plot_fit:
        fit = kink_density_theory(tau, a, e, g)
        figure, ax = plt.subplots(2, figsize=(8, 6), gridspec_kw={'height_ratios': [2, 1]})
        ax[1].set_xlabel(xlabel or '$\\tau_Q$')
        ax[0].set_ylabel('Kink density')
        ax[1].set_ylabel("Residuals")
        ax[0].plot(tau, kdens_mean)
        ax[0].plot(tau[:data_end], fit[:data_end],
                   label=f"$f(\\tau)={round(a, 2)} \\cdot \\tau ** ({round(e, 2)}) * e**(-\\tau_Q*{round(g, 2)})$")
        ax[0].plot(tau[:data_end], kink_density_theory(tau, 1 / (2 * np.pi * np.sqrt(2)), -0.5, g)[:data_end],
                   label="Theory", linestyle="--")
        ax[1].plot(tau[data_start:data_end], (kdens_mean - fit)[data_start:data_end], label="Data - fit")
        ax[1].axhline(0, color="red", linestyle="--")
        ax[0].set_yscale("log")
        ax[0].set_xscale("log")
        ax[1].set_xscale("log")

        ax[0].legend()
        ax[1].legend()
        ax[0].grid()
        ax[1].grid()
        plt.tight_layout()
        plt.savefig("../figs/" + filename + ".png")
    else:
        plt.plot(tau, kdens_mean)
        plt.fill_between(tau, kdens_mean - kdens_sig, kdens_mean + kdens_sig, alpha=0.2)
        plt.xlabel(xlabel or '$\\tau_Q$')
        plt.ylabel('Kink density')
        plt.yscale("log")
        plt.xscale("log")
        plt.legend()
        plt.tight_layout()
        plt.savefig("../figs/" + filename + ".png")


def plot_fidelities(dt, steps, fidelities, filename):
    steps_arr = np.array(range(1, steps + 1)) / steps
    for i, fids in enumerate(fidelities):
        plt.plot(steps_arr, fids, label=f"$\tau_Q$={steps * dt[i]}")
        plt.xlabel('$t/\\tau_Q$')
        plt.ylabel('Fidelity')
        plt.legend()
        plt.tight_layout()
        plt.savefig("../figs/" + filename + ".png")


def kink_density_theory(tau, a, e, g):
    return a * tau ** e * np.exp(-tau * g)


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
                        help="0 for kink density estimation, 1 for plain evolution of a given system, 2 for kink density magnetic field estimation and 3 for fidelity estimation",
                        type=int,
                        default=0)
    parser.add_argument("-gpu", "--gpu", help="1 to use GPU, else CPU", type=int, default=0)
    parser.add_argument("-s", "--samples", help="Number of sample states to get from one execution of the circuit",
                        type=int,
                        default=1)
    parser.add_argument("-p", "--periodic", help="1 for Periodic boundary conditions else Open", type=int,
                        default=0)
    parser.add_argument("-inv", "--inverse", help="1 for inverse magnetic field ramp h_0 -> 0 else 0 -> h_0", type=int,
                        default=0)
    parser.add_argument("-b", "--bias", help="Bias for the parallel magnetic field. default: None", type=float,
                        default=None)

    args = parser.parse_args()

    N_ = args.N
    trotter_steps_ = args.trotter_steps
    J_ = args.J_values
    h_ = args.h_values
    dt_ = args.dt  # time step
    obs_Z_ = [5, 6]
    obs_XX_ = [[4, 14], [5, 6]]
    data_start = 7
    data_end = 200
    gpu_ = bool(args.gpu)
    samples_ = args.samples
    mode = args.mode
    periodic = bool(args.periodic)
    inverse = bool(args.inverse)
    bias = args.bias

    if mode == 0:
        estimate_kinks_tau_dependency(N_, dt_, h_, J_, trotter_steps_, gpu=gpu_, samples=samples_,
                                      periodic=periodic, inverse=inverse, bias=bias)

        filename = f"kinks_N{N_}_J{J_}_h{h_}_dt{dt_}_steps{trotter_steps_}_periodic{periodic}_inv{inverse}_bias{bias}"
        tau, kinks_mean, kinks_sig = np.array(load_file("../data/" + filename))
        popt, pcov = fit_kinks(tau[data_start:data_end], kinks_mean[data_start:data_end],
                               kinks_sig[data_start:data_end] + 1e-15)
        plot_dependency("../figs/" + filename, tau, kinks_mean, kinks_sig, plot_fit=True, a=popt[0], e=popt[1],
                        g=popt[2], data_start=data_start, data_end=data_end)
    elif mode == 1:
        evolve_system(N_, dt_, h_, J_, trotter_steps_, obs_Z_, obs_XX_, gpu=bool(args.gpu), samples=samples_,
                      periodic=periodic, inverse=inverse, bias=bias)
    elif mode == 2:
        estimate_kinks_magnetic_dependency(N_, dt_, h_, J_, trotter_steps_, gpu=bool(args.gpu), samples=samples_,
                                           periodic=periodic, inverse=inverse)
    elif mode == 3:
        dt = np.array([0.00001, 0.0001, 0.001, 0.01, 0.1])
        fidelity_measurement(N_, h_, J_, trotter_steps_, dt, gpu_, periodic, inverse, bias)
        filename = f'fidelity_N{N_}_J{J_}_h{h_}_steps{trotter_steps_}_periodic{periodic}_inv{inverse}_bias{bias}'
        steps, fidelities, _sig = load_file("../data/" + filename)
        plot_fidelities(dt, steps, fidelities, filename)

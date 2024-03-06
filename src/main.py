import argparse
import numpy as np
from measurements import kinks_time_measurements, observable_measurements, kinks_magneticfield_measurements, \
    fidelity_measurements, iteration_state_evolution_parallel, correlator_measurements, entropy_measurements
from data_processing import correlator_processing, entropies_processing
from filemanager import load_file
from plotting import plot_kinks, plot_fidelities, plot_correlators_time, plot_correlators_distance, \
    plot_entropies_time, plot_entropies_distance
from theory import fit_kinks

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
    data_start = 3
    data_end = 15
    gpu_ = bool(args.gpu)
    samples_ = args.samples
    mode = args.mode
    periodic = bool(args.periodic)
    inverse = bool(args.inverse)
    bias = args.bias

    # Kinks - quenchtime dependency
    if mode == 0:
        kinks_time_measurements(N_, dt_, h_, J_, trotter_steps_, gpu=gpu_, samples=samples_,
                                periodic=periodic, inverse=inverse, bias=bias)

        filename = f"kinks_N{N_}_J{J_}_h{h_}_dt{dt_}_steps{trotter_steps_}_periodic{periodic}_inv{inverse}_bias{bias}"
        tau, kinks_mean, kinks_var, kinks_skewness = np.array(load_file("../data/kinks/" + filename))
        popt, pcov = fit_kinks(tau[data_start:data_end], kinks_mean[data_start:data_end])
        plot_kinks(filename, tau, kinks_mean, kinks_var, kinks_skewness, plot_fit=True, a=popt[0], e=popt[1], g=popt[2],
                   data_start=data_start, data_end=data_end)
    # Evolution of system and measuring Correlators of a single run
    elif mode == 1:
        observable_measurements(N_, dt_, h_, J_, trotter_steps_, obs_Z_, obs_XX_, gpu=bool(args.gpu), samples=samples_,
                                periodic=periodic, inverse=inverse, bias=bias)
    # Kinks - Magnetic field dependency
    elif mode == 2:
        kinks_magneticfield_measurements(N_, dt_, h_, J_, trotter_steps_, gpu=bool(args.gpu), samples=samples_,
                                         periodic=periodic, inverse=inverse)
    # Fidelity in dependence of evolution step
    elif mode == 3:
        dt = np.array([0.00001, 0.0001, 0.001, 0.01, 0.1])
        fidelity_measurements(N_, h_, J_, trotter_steps_, dt, gpu_, periodic, inverse, bias)
        filename = f'fidelity_N{N_}_J{J_}_h{h_}_steps{trotter_steps_}_periodic{periodic}_inv{inverse}_bias{bias}'
        steps, fidelities, _sig = load_file("../data/fidelities/" + filename)
        plot_fidelities(dt, steps, fidelities, filename)
    # Evolution of system and measuring Correlators for several runs with different quench times
    elif mode == 4:
        dt = np.array([0.5, 1, 2, 4, 8, 16, 32]) / trotter_steps_

        iteration_state_evolution_parallel(N_, h_, J_, trotter_steps_, dt, gpu=gpu_, periodic=periodic, inverse=inverse,
                                           bias=bias)
        filename = f'states_N{N_}_J{J_}_h{h_}_steps{trotter_steps_}_dt{dt}_periodic{periodic}_inv{inverse}_bias{bias}'
        _, quench_runs, _, _ = load_file("../data/states/" + filename)

        filename = f"correlators_N{N_}_J{J_}_h{h_}_steps{trotter_steps_}_periodic{periodic}_inv{inverse}_bias{bias}"
        correlator_measurements(N_, quench_runs, filename)
        tau, correlators, _, _ = load_file("../data/correlators/" + filename)

        processed_correlators = correlator_processing(N_, correlators, periodic=periodic)
        plot_correlators_time(processed_correlators, filename)
        plot_correlators_distance(processed_correlators, filename, critical=True)
        plot_correlators_distance(processed_correlators, filename, critical=False)
    elif mode == 5:
        dt = np.array([0.5, 1, 2, 4, 8, 16, 32]) / trotter_steps_
        iteration_state_evolution_parallel(N_, h_, J_, trotter_steps_, dt, gpu=gpu_, periodic=periodic, inverse=inverse,
                                           bias=bias)
        filename = f'states_N{N_}_J{J_}_h{h_}_steps{trotter_steps_}_dt{dt}_periodic{periodic}_inv{inverse}_bias{bias}'
        _, quench_runs, _, _ = load_file("../data/states/" + filename)

        filename = f'entropies_N{N_}_J{J_}_h{h_}_steps{trotter_steps_}_dt{dt}_periodic{periodic}_inv{inverse}_bias{bias}'
        entropy_measurements(N_, quench_runs, filename)
        _, entropies, _, _ = load_file("../data/entropies/" + filename)

        distance_entropies = entropies_processing(N_, entropies['2s'], periodic=periodic)
        plot_entropies_time(entropies['3s'], filename)
        plot_entropies_distance(distance_entropies, filename)

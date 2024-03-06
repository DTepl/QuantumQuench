from itertools import repeat
import numpy as np
from IsingEvolution import IsingEvol
from multiprocessing import Pool
from plotting import plot_kinks
from filemanager import save_file


def observable_measurements(N, dt, h, J, trotter_steps, obs_Z, obs_XX, gpu=False, periodic=False, samples=1,
                            inverse=False, bias=None):
    # Initialization
    ising_model_evolution = IsingEvol(N, dt, h, J, gpu=gpu, inverse=inverse, bias=bias, periodic=periodic)
    ising_model_evolution.observables([], obs_Z, obs_XX, [])

    # Execution
    states = ising_model_evolution.execute(draw=False, steps=trotter_steps, samples=samples)
    expectations = ising_model_evolution.compute_expectationvals(states)
    ising_model_evolution.plot(expectations)


def iteration_kinks(ising_model_evolution, steps, samples, h=None):
    states = ising_model_evolution.execute(draw=False, steps=steps, samples=samples, h=h)
    nok_mean, nok_var, nok_skewness = ising_model_evolution.compute_kink_density(states)
    return [nok_mean, nok_var, nok_skewness]


def iteration_fidelity_groundstate_dot(ising_obj: IsingEvol, state: list, step: int, steps: int):
    groundstate = ising_obj.groundState(step, steps)
    return (np.abs(np.conj(state) @ groundstate) ** 2).flatten()


def iteration_state_evolution(N, h, J, dt, steps, samples, periodic, inverse, bias, gpu):
    ising_model_evolution = IsingEvol(N, dt, h, J, gpu=gpu, periodic=periodic, inverse=inverse, bias=bias)
    ising_model_evolution.progress = False
    states = ising_model_evolution.execute(draw=False, steps=steps, samples=samples)

    arr = []
    for step in range(1, len(states) + 1):
        arr.append(states[str(step)].data)

    return np.array(arr)


def iteration_state_evolution_parallel(N, h, J, steps, dt, gpu=False, periodic=False, inverse=False, bias=None):
    params = zip(repeat(N), repeat(h), repeat(J), dt, repeat(steps), repeat(steps), repeat(periodic), repeat(inverse),
                 repeat(bias), repeat(gpu))

    with Pool() as pool:
        states = np.array(pool.starmap(iteration_state_evolution, params))

    result = {}
    for count, quench in enumerate(states):
        result[str(dt[count] * steps)] = quench

    filename = f'states/states_N{N}_J{J}_h{h}_steps{steps}_dt{dt}_periodic{periodic}_inv{inverse}_bias{bias}'
    things_to_save = [dt, result, None, None]
    save_file(filename, things_to_save)


def iteration_entropy(partial_trace_set: list[np.ndarray]):
    res = []
    for partial_trace in partial_trace_set:
        res.append(-np.sum(np.real(np.trace(partial_trace * np.log2(partial_trace)))))

    return res


def kinks_time_measurements(N, dt, h, J, trotter_steps, gpu=False, samples=1, periodic=False, inverse=False,
                            bias=None):
    ising_model_evolution = IsingEvol(N, dt, h, J, gpu=gpu, periodic=periodic, inverse=inverse, bias=bias)
    ising_model_evolution.progress = False
    steps = range(1, trotter_steps + 1)
    tau = dt * np.array(steps)

    with Pool() as pool:
        nok = np.array(
            pool.starmap(iteration_kinks, zip(repeat(ising_model_evolution), steps, repeat(samples), repeat(None))))

    filename = f'kinks/kinks_N{N}_J{J}_h{h}_dt{dt}_steps{trotter_steps}_periodic{periodic}_inv{inverse}_bias{bias}'
    kdens_exp = nok[:, 0]
    kdens_var = nok[:, 1]
    kdens_skew = nok[:, 2]
    things_to_save = [tau, kdens_exp, kdens_var, kdens_skew]
    save_file(filename, things_to_save)


def kinks_magneticfield_measurements(N, dt, h, J, trotter_steps, gpu=False, samples=1, periodic=False, inverse=False,
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
    plot_kinks(filename, np.array(h_array), np.array(kdens_mean), np.array(kdens_sig), xlabel="Magnetic field h")
    # plot_dependency(filename, np.array(h_array), np.array(nok), np.array(trotter_steps * [0]),
    #                 xlabel="Magnetic field h")
    things_to_save = [h_array, np.array(nok), np.array(trotter_steps * [0])]
    save_file(filename, things_to_save)


def fidelity_measurements(N, h, J, steps, dt, gpu=False, periodic=False, inverse=False, bias=None):
    params = zip(repeat(N), repeat(h), repeat(J), dt, repeat(steps), repeat(steps), repeat(periodic), repeat(inverse),
                 repeat(bias), repeat(gpu))

    with Pool() as pool:
        states = np.array(pool.starmap(iteration_state_evolution, params))

        groundstate_computer = IsingEvol(N, 0, h, J, gpu=gpu, periodic=periodic, inverse=inverse, bias=bias)
        # Swap axes from (5,100,-1) to (100,5,-1) where 5 are the number of dt's and 100 are the steps
        fidelities = pool.starmap(iteration_fidelity_groundstate_dot,
                                  zip(repeat(groundstate_computer), np.swapaxes(states, 0, 1), range(1, steps + 1),
                                      repeat(steps)))

    filename = f'fidelities/fidelity_N{N}_J{J}_h{h}_steps{steps}_periodic{periodic}_inv{inverse}_bias{bias}'
    things_to_save = [steps, np.swapaxes(fidelities, 0, 1), 0]  # swap back for easier plotting
    save_file(filename, things_to_save)


def correlator_measurements(N: int, quench_runs: dict, filename: str):
    obs_idx = {
        'X': [[i] for i in range(N)],
        'Z': [[i] for i in range(N)],
        'XX': [[x, y] for x in range(N) for y in range(x + 1, N)],
        'ZZ': [[x, y] for x in range(N) for y in range(x + 1, N)]
    }

    obs = IsingEvol.observables(N, obs_idx)

    with Pool() as pool:
        expectation_value = pool.starmap(IsingEvol.compute_expectationvals,
                                         zip(repeat(obs), list(quench_runs.values())))

    correlators = {}
    keys = list(quench_runs.keys())
    for count, quench_run in enumerate(expectation_value):
        correlator_iter = {}
        for obsstr in obs_idx:
            correlator_iter[obsstr] = quench_run[obsstr].copy()

            if len(obsstr) == 2:
                for idx in quench_run[obsstr]:
                    correlator_iter[obsstr][idx] = quench_run[obsstr][idx] - expectation_value[count][obsstr[0]][
                        tuple([idx[0]])] * expectation_value[count][obsstr[1]][tuple([idx[1]])]

        correlators[keys[count]] = correlator_iter

    things_to_save = [keys, correlators, None, None]
    save_file('correlators/' + filename, things_to_save)


def entropy_measurements(N: int, quench_runs: dict[list], filename: str):
    midpoint = int((N - 1) / 2)
    keepsub_2s = [[x, y] for x in range(N) for y in range(x + 1, N)]
    keepsub_3s = [[midpoint - i, midpoint, midpoint + i] for i in range(1, min(midpoint, 3) + 1)]
    entropies_set = {'2s': {}, '3s': {}}

    for quench_time in quench_runs:
        states = quench_runs[quench_time]
        params_2s = zip(repeat(N), repeat(states), keepsub_2s)
        params_3s = zip(repeat(N), repeat(states), keepsub_3s)

        with Pool() as pool:
            partial_traces_2s = pool.starmap(IsingEvol.compute_partial_trace, params_2s)
            entropies_2s = {tuple(keepsub_2s[counter]): np.array(entropies_set) for counter, entropies_set in
                            enumerate(pool.map(iteration_entropy, partial_traces_2s))}

            partial_traces_3s = pool.starmap(IsingEvol.compute_partial_trace, params_3s)
            entropies_3s = {tuple(keepsub_3s[counter]): np.array(entropies_set) for counter, entropies_set in
                            enumerate(pool.map(iteration_entropy, partial_traces_3s))}

        entropies_set['2s'][quench_time] = entropies_2s
        entropies_set['3s'][quench_time] = entropies_3s

    things_to_save = [list(quench_runs.keys()), entropies_set, None, None]
    save_file('entropies/' + filename, things_to_save)

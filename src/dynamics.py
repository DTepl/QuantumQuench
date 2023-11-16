import numpy as np

from src.IsingEvolution import IsingEvol
import matplotlib.pyplot as plt
import tqdm

N_ = 20
trotter_steps_ = 100
J_ = -0.25
h_ = -1.5
dt_ = 0.1  # time step
obs_Z_ = [5, 6]
obs_XX_ = [[4, 14], [5, 6]]


def evolve_system(N, dt, h, J, trotter_steps, obs_Z, obs_XX, linear_increase=True):
    ising_model_evolution = IsingEvol(N, dt, h, J)
    ising_model_evolution.observables(obs_Z, obs_XX)
    ising_model_evolution.execute(draw=False, steps=trotter_steps, linear_increase=linear_increase)
    ising_model_evolution.compute_expectationvals()
    ising_model_evolution.plot()


def estimate_kinks_tau_dependency(N, dt, h, J, trotter_steps):
    ising_model_evolution = IsingEvol(N, dt, h, J)
    ising_model_evolution.progress = False
    tau = []
    kinks_mean = []
    kinks_sig = []

    for step in tqdm.tqdm(range(1, trotter_steps + 1)):
        tau.append(dt * step)
        ising_model_evolution.execute(draw=False, steps=step)
        ising_model_evolution.compute_kink_density()
        kinks_mean.append(ising_model_evolution.nok[0])
        kinks_sig.append(ising_model_evolution.nok[1])

    plt.plot(np.array(tau), np.array(kinks_mean))
    plt.xlabel('$\\tau_Q$')
    plt.ylabel('Kinks')
    plt.savefig(f'../figs/kinks_N{N}_J{J}_h{h}_dt{dt}_steps{trotter_steps}.png')


estimate_kinks_tau_dependency(N_, dt_, h_, J_, trotter_steps_)

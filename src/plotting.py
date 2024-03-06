import matplotlib.pyplot as plt
import numpy as np

from src.theory import kink_density_theory


def plot_kinks(filename, tau, kdens_mean, kdens_var, kdens_skewness, plot_fit=False, a=0, e=0, g=0, xlabel=None,
               data_start=0, data_end=-1):
    plt.plot(tau, kdens_mean)
    print(f"exponent: {e}")
    print(f"parallel magnetic: {(g / 6.89) ** (15 / 16)}")
    if plot_fit:
        fit = kink_density_theory(tau, a, e, g)
        figure, ax = plt.subplots(2, figsize=(8, 6), gridspec_kw={'height_ratios': [2, 1]})
        ax[0].set_ylabel('Observables')
        ax[0].plot(tau, kdens_mean, label="$\\left<N\\right>$")
        ax[0].plot(tau, kdens_var, label="$\\left<N^2\\right> - \\left<N\\right>^2$")
        ax[0].plot(tau, kdens_skewness, label="$\\left<(N-\\left<N\\right>)^3\\right>$")
        ax[0].plot(tau[:data_end], fit[:data_end],
                   label=f"$f(\\tau)={round(a, 2)} \\cdot \\tau ** ({round(e, 2)}) * e**(-\\tau_Q*{round(g, 2)})$")
        ax[0].plot(tau[:data_end], kink_density_theory(tau, 1 / (2 * np.pi * np.sqrt(2)), -0.5, g)[:data_end],
                   label="Theory", linestyle="--")
        ax[0].set_yscale("log")
        ax[0].set_xscale("log")
        ax[0].legend()
        ax[0].grid()

        ax[1].set_xlabel(xlabel or '$\\tau_Q$')
        ax[1].set_ylabel("Residuals")
        ax[1].set_xscale("log")
        ax[1].plot(tau[data_start:data_end], (kdens_mean - fit)[data_start:data_end], label="Data - fit")
        ax[1].axhline(0, color="red", linestyle="--")
        ax[1].grid()
        ax[1].legend()

        plt.tight_layout()
        plt.savefig("../figs/kinks/" + filename + ".png")
    else:
        plt.plot(tau, kdens_mean, label="$\\left<N\\right>$")
        plt.plot(tau, kdens_var, label="$\\left<N^2\\right> - \\left<N\\right>^2$")
        plt.plot(tau, kdens_skewness, label="$\\left<(N-\\left<N\\right>)^3\\right>$")
        plt.xlabel(xlabel or '$\\tau_Q$')
        plt.ylabel('Observables')
        plt.yscale("log")
        plt.xscale("log")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig("../figs/kinks/" + filename + ".png")


def plot_fidelities(dt, steps, fidelities, filename):
    steps_arr = np.array(range(1, steps + 1)) / steps
    for i, fids in enumerate(fidelities):
        plt.plot(steps_arr, fids, label=f"$\\tau_Q$={steps * dt[i]}")
        plt.xlabel('$t/\\tau_Q$')
        plt.ylabel('Fidelity')
        plt.legend()
        plt.tight_layout()
        plt.savefig("../figs/fidelities/" + filename + ".png")


def plot_correlators_time(correlators, filename):
    figure_time, ax_time = plt.subplots(4, figsize=(8, 12), gridspec_kw={'height_ratios': [3, 2, 2, 2]})
    N = len((list(correlators.values())[0])['Z'])
    close_bound = 1

    for tau in correlators:
        data_ZZ = np.array(correlators[tau]['ZZ'][tuple([1])])
        x = np.linspace(1 / data_ZZ.shape[1], 1, data_ZZ.shape[1])
        y_ZZ = [np.mean(data_ZZ[:, i]) for i in range(data_ZZ.shape[1])]
        ax_time[0].plot(x, y_ZZ, label=f"$\\tau_Q$={tau}")

        data_X = []
        for idx in correlators[tau]['X']:
            data_X.append(correlators[tau]['X'][idx])
        data_X = np.array(data_X)
        y_X = [np.mean(data_X[:, i]) for i in range(data_X.shape[1])]
        ax_time[3].plot(x, y_X, label=f"tau={tau}")

        data_Z = np.array(correlators[tau]['Z'][tuple([close_bound])])
        ax_time[2].plot(x, data_Z, label=f"tau={tau}")

        data_Z = np.array(correlators[tau]['Z'][tuple([int(N / 2) - 1])])
        ax_time[1].plot(x, data_Z, label=f"tau={tau}")

    ax_time[0].set_ylabel('$C^{ZZ}_1$')
    ax_time[0].legend()
    ax_time[0].axhline(0, color="red", linestyle="--")
    ax_time[0].grid()

    ax_time[3].set_ylabel("$\left<\sigma^X\\right>$")
    ax_time[3].axhline(0, color="red", linestyle="--")
    ax_time[3].grid()
    ax_time[3].legend()
    ax_time[3].set_xlabel('$t/\\tau_Q$')

    ax_time[2].set_ylabel(r"$\left<\sigma^Z_{{{}}}\right>$".format(close_bound))
    ax_time[2].axhline(0, color="red", linestyle="--")
    ax_time[2].grid()
    ax_time[2].legend()

    ax_time[1].set_ylabel(r"$\left<\sigma^Z_{{{}}}\right>$".format(int(N / 2) - 1))
    ax_time[1].axhline(0, color="red", linestyle="--")
    ax_time[1].grid()
    ax_time[1].legend()

    figure_time.tight_layout()
    figure_time.savefig("../figs/correlators/" + filename + "_time.png")
    figure_time.clf()


def plot_correlators_distance(correlators, filename, critical=False):
    figure_distance, ax_distance = plt.subplots(3, figsize=(8, 12), gridspec_kw={'height_ratios': [3, 2, 2]})

    for tau in correlators:
        distances_XX = []
        distances_ZZ = []
        y_ZZ = []
        y_XX = []

        for distance in correlators[tau]['ZZ']:
            distances_ZZ.append(distance[0])
            data_ZZ = np.array(correlators[tau]['ZZ'][distance])
            y_ZZ.append(np.mean(data_ZZ[:, (int(data_ZZ.shape[1] / 2) - 1 if critical else 0)]))

        for distance in correlators[tau]['XX']:
            distances_XX.append(distance[0])
            data_XX = np.array(correlators[tau]['XX'][distance])
            y_XX.append(np.mean(data_XX[:, (int(data_XX.shape[1] / 2) - 1 if critical else 0)]))

        ax_distance[0].plot(distances_ZZ, y_ZZ, label=f"$\\tau_Q$={tau}")
        ax_distance[1].plot(distances_XX, y_XX, label=f"$\\tau_Q$={tau}")
        ax_distance[2].plot(np.array(distances_XX) / np.sqrt(float(tau)), y_XX, label=f"$\\tau_Q$={tau}")

    ax_distance[0].set_ylabel('$C^{ZZ}_R$')
    ax_distance[0].legend()
    ax_distance[0].set_yscale("log")
    ax_distance[0].grid()

    ax_distance[1].set_ylabel('$C^{XX}_R$')
    ax_distance[1].grid()
    ax_distance[1].legend()
    ax_distance[1].set_xlabel('R')

    ax_distance[2].set_ylabel('$C^{XX}_R$')
    ax_distance[2].grid()
    ax_distance[2].legend()
    ax_distance[2].set_xlabel('$R/(\\tau_Q)^{1/2}$')

    figure_distance.tight_layout()
    figure_distance.savefig("../figs/correlators/" + filename + f"_critical{critical}.png")
    figure_distance.clf()


def plot_entropies_time(entropies_set: dict, filename: str):
    figure_time, ax_time = plt.subplots(len(list(entropies_set.values())[0]), figsize=(8, 12))

    for tau in entropies_set:
        for count, idx in enumerate(entropies_set[tau]):
            data = entropies_set[tau][idx]
            x = np.linspace(1 / data.shape[0], 1, data.shape[0])
            ax_time[count].plot(x, data, label=f"tau={tau}")

            ax_time[count].set_ylabel(r'$S_{{{}}}$'.format(idx))
            ax_time[count].legend()
            ax_time[count].grid()

    ax_time[-1].set_xlabel('$t/\\tau_Q$')
    figure_time.tight_layout()
    figure_time.savefig("../figs/entropies/" + filename + "_time.png")
    figure_time.clf()


def plot_entropies_distance(entropies_set, filename):
    figure_distance, ax_distance = plt.subplots(2, 2, figsize=(10, 8))

    for tau in entropies_set:
        distances = []
        y_init = []
        y_crit = []

        for distance in entropies_set[tau]:
            distances.append(distance[0])
            data = np.array(entropies_set[tau][distance])
            y_init.append(np.mean(data[:, 0]))
            y_crit.append(np.mean(data[:, int(data.shape[1] / 2) - 1]))
        y_init = np.array(y_init)
        y_crit = np.array(y_crit)
        distances = np.array(distances)

        ax_distance[0, 0].plot(distances, y_init, label=f"$\\tau_Q$={tau}")
        ax_distance[1, 0].plot(distances, y_crit, label=f"$\\tau_Q$={tau}")
        ax_distance[0, 1].plot(distances / np.sqrt(float(tau)), y_init, label=f"$\\tau_Q$={tau}")
        ax_distance[1, 1].plot(distances / np.sqrt(float(tau)), y_crit, label=f"$\\tau_Q$={tau}")

    ax_distance[0, 0].set_ylabel('$S_{init}$')
    ax_distance[0, 0].legend()
    # ax_distance[0, 0].set_yscale("log")
    ax_distance[0, 0].grid()

    # ax_distance[0, 1].set_ylabel('$S_{VN}$')
    ax_distance[0, 1].grid()
    ax_distance[0, 1].legend()
    # ax_distance[0, 1].set_xlabel('R')

    ax_distance[1, 0].set_ylabel('$S_{crit}$')
    ax_distance[1, 0].legend()
    # ax_distance[0, 0].set_yscale("log")
    ax_distance[1, 0].grid()
    ax_distance[1, 0].set_xlabel('R')

    # ax_distance[1, 1].set_ylabel('$S_{VN}$')
    ax_distance[1, 1].legend()
    # ax_distance[0, 0].set_yscale("log")
    ax_distance[1, 1].grid()
    ax_distance[1, 1].set_xlabel('$R/(\\tau_Q)^{1/2}$')

    figure_distance.tight_layout()
    figure_distance.savefig("../figs/entropies/" + filename + "_dist.png")
    figure_distance.clf()

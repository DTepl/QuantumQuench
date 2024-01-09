import numpy as np
import logging as log
import matplotlib.pyplot as plt
from multiprocessing import Pool
import pickle
from scipy.optimize import curve_fit
from qutip import basis, qeye, sigmax, sigmaz, tensor, Options, sesolve

log.basicConfig(level=log.INFO)


class AnalyticalIsing:
    def __init__(self, N, J, h, periodic=False):
        self.J = J
        self.h = h
        self.N = N
        self.periodic = periodic
        self.obs = self.observable()
        self.hamiltonian_comps = self.get_hamiltonian_components()
        self.H_t = [self.hamiltonian_comps[0], [self.hamiltonian_comps[1], f'{self.h}*(1 - (t/tq))']]

    def plot(self, tau, kink_density, A=0, e=0, g=0):
        plt.plot(tau, kink_density)
        plt.plot(tau, self.kink_density_theory(tau, A, e, g), label=f"{round(A, 2)}*tau^{round(e, 2)}*e^(-tau * {round(g, 2)})")
        plt.xlabel('$\\tau_Q$')
        plt.ylabel('Kink density')
        plt.yscale("log")
        plt.xscale("log")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(
            f"../figs/AnalyticalSolution_N{self.N}_J{self.J}_h{self.h}_tmax{int(tau[-1])}_num{len(tau)}_periodic{self.periodic}.png")

    def kink_density_theory(self, tau, a, e, g):
        return a * tau ** e #* np.exp(-tau * g)

    def get_hamiltonian_components(self):
        # Setup operators for individual qubits
        sx_list, sz_list = [], []
        for i in range(self.N):
            op_list = [qeye(2)] * self.N
            op_list[i] = sigmax()
            sx_list.append(tensor(op_list))
            op_list[i] = sigmaz()
            sz_list.append(tensor(op_list))

        # Hamiltonian - Energy splitting terms
        H_ext = 0
        for i in range(self.N):
            H_ext += sz_list[i]

        H_int = 0
        # Interaction terms
        for n in range(self.N - 1):
            H_int += self.J * sx_list[n] * sx_list[n + 1]

        if self.periodic:
            H_int += self.J * sx_list[0] * sx_list[self.N - 1]

        return (H_int, H_ext)

    def get_groundstate(self, H):
        return H.groundstate()

    def observable(self):
        log.info(f"Computing Observable")
        obs = tensor([qeye(2)] * self.N) * (self.N - (0 if self.periodic else 1))

        sx_list = []
        for i in range(self.N):
            op_list = [qeye(2)] * self.N
            op_list[i] = sigmax()
            sx_list.append(tensor(op_list))

        for n in range(self.N - 1):
            obs -= sx_list[n] * sx_list[n + 1]

        if self.periodic:
            obs -= sx_list[0] * sx_list[self.N - 1]

        log.info(f"Observable computed")
        return obs / (2 * self.N)

    def evolve_state(self, H, psi0, exp, times, args):
        return sesolve(H, psi0, times, e_ops=exp, args=args, options=Options(rhs_reuse=True, nsteps=5000)).expect[0][-1]

    def _evolve_state(self, tau):
        # tau, step = tau_step
        times = np.linspace(tau / 10, tau, 10)
        args = {'tq': tau}
        return self.evolve_state(self.H_t, tensor([basis(2, 1)] * self.N), [self.obs], times,
                                 args)

    def tau_sweep(self, t_max, resolution):
        log.info(f"Doing tauQ sweep for max tq max {t_max} and {resolution} number of steps")
        tau = np.linspace(t_max / resolution, t_max, resolution)

        with Pool() as pool:
            nok = np.array(pool.map(self._evolve_state, tau))
            # nok = np.array(pool.map(self._evolve_state, zip(repeat(tau), range(resolution))))

        log.info(f"TauQ sweep done!")

        filename = f'AnalyticalSolution_N{self.N}_J{self.J}_h{self.h}_tmax{t_max}_num{resolution}_periodic{self.periodic}'

        things_to_save = [tau, nok]

        with open("../data/" + filename, "wb") as f:
            pickle.dump(things_to_save, f)

        return tau, nok


if __name__ == '__main__':
    res = 1000
    start = 1
    end = 110
    t_max = 100
    periodic = False

    anaIsing = AnalyticalIsing(12, -1, 2, periodic=periodic)
    tau, kink_density = anaIsing.tau_sweep(t_max, res)

    filename = f"../data/AnalyticalSolution_N{anaIsing.N}_J{anaIsing.J}_h{anaIsing.h}_tmax{t_max}_num{res}_periodic{periodic}"
    with open(filename, "rb") as f:
        things_to_load = pickle.load(f)
        [tau, kink_density] = things_to_load

    popt, pcov = curve_fit(anaIsing.kink_density_theory, tau[start:end], kink_density[start:end])
    anaIsing.plot(tau, kink_density, popt[0], popt[1], popt[2])

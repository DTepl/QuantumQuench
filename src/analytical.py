from itertools import repeat

from scipy.sparse.linalg import eigsh, expm
import numpy as np
import logging as log
import matplotlib.pyplot as plt
from multiprocessing import Pool
import pickle
from scipy.optimize import curve_fit
from qutip import about, basis, expect, mesolve, qeye, sigmax, mcsolve, sigmaz, tensor, brmesolve, Options, sesolve

from qiskit.quantum_info.operators import SparsePauliOp

log.basicConfig(level=log.INFO)


class AnalyticalIsing:
    def __init__(self, dt, N, J, h, periodic=False):
        self.J = J
        self.h = h
        self.N = N
        self.dt = dt
        self.periodic = periodic
        # self.interaction, self.external = self.basis()
        # self.groundstate = self.groundstate()
        self.obs = self.observable()
        self.hamiltonian_comps = self.get_hamiltonian_components()
        self.H_t = [self.hamiltonian_comps[0], [self.hamiltonian_comps[1], f'{self.h}*(1 - (t/tq))']]

    def basis(self):
        log.info(f"Computing Hamiltonian basis")
        interaction = []
        external = []

        for i in range(self.N - 1):
            interaction.append(('XX', [i, i + 1], self.J))

        if self.periodic:
            interaction.append(('XX', [0, self.N - 1], self.J))

        for i in range(self.N):
            external.append(('Z', [i], 1))

        interaction_matr = SparsePauliOp.from_sparse_list(interaction, num_qubits=self.N).to_matrix(sparse=True).tocsc()
        external_matr = SparsePauliOp.from_sparse_list(external, num_qubits=self.N).to_matrix(sparse=True).tocsc()
        log.info(f"Computed hamiltonian basis")
        return interaction_matr, external_matr

    def hamiltonian(self, proportion):
        return self.interaction + self.h * (1 - proportion) * self.external

    def groundstate(self):
        log.info(f"Computing groundstate of hamiltonian")
        eigval, eigvec = eigsh(self.hamiltonian(0), which="SA", k=1)
        log.info(f"Computed groundstate with eigenvalue {eigval}")
        return eigvec.flatten()

    def quench(self, steps):
        evol = self.groundstate
        for i in range(1, steps + 1):
            evol = expm(-1j * self.hamiltonian(i / steps) * self.dt) @ evol

        log.info(f"Computed evolution operator for tq={self.dt * steps}")
        log.info(f"Computing expectation values")
        res_exp = np.abs(np.conj(evol) @ self.observable @ evol)
        log.info(f"Expectation values computed!")
        return res_exp

    # def observable(self):
    #     log.info(f"Computing Observable")
    #     result = [('I', [1], self.N - (0 if self.periodic else 1))]
    #
    #     for i in range(self.N - 1):
    #         result.append(('XX', [i, i + 1], -1))
    #
    #     if self.periodic:
    #         result.append(('XX', [0, self.N - 1], -1))
    #
    #     matr = SparsePauliOp.from_sparse_list(result, num_qubits=self.N).to_matrix(sparse=True) / (2 * self.N)
    #     log.info(f"Observable computed")
    #     return matr

    # def tau_sweep(self, resolution):
    #     log.info(f"Doing tauQ sweep for max tq {self.dt * resolution} and {resolution} number of steps")
    #     steps = range(1, resolution + 1)
    #
    #     with Pool() as pool:
    #         nok = np.array(pool.map(self.quench, steps))
    #
    #     log.info(f"TauQ sweep done!")
    #
    #     filename = f'AnalyticalSolution_N{self.N}_J{self.J}_h{self.h}_dt{self.dt}_num{resolution}'
    #     things_to_save = [np.array(steps) * self.dt, nok]
    #
    #     with open("../data/" + filename, "wb") as f:
    #         pickle.dump(things_to_save, f)
    #
    #     return np.array(steps) * self.dt, nok

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
        return a * tau ** e * np.exp(-tau * g)

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
        return sesolve(H, psi0, times, e_ops=exp, args=args, options=Options(rhs_reuse=True)).expect[0][-1]

    def _evolve_state(self, tau):
        # tau, step = tau_step
        times = np.linspace(tau / 10, tau, 10)
        args = {'tq': tau}
        return self.evolve_state(self.H_t, tensor([basis(2, 1)] * self.N), [self.obs], times,
                                 args)

    def tau_sweep(self, t_max, resolution):
        tau = np.linspace(t_max / resolution, t_max, resolution)
        # eigen, groundstate = self.get_groundstate(hamiltonian_comps[0] + self.h * hamiltonian_comps[1])
        # res = self.evolve_state(H_t,groundstate, [observable], tau, args)

        # length = len(tau) - 1
        # for i in range(len(tau)):
        #     args = {'tq': tau[i]}
        #     res = self.evolve_state(self.H_t, tensor([basis(2, 1)] * self.N), [self.obs], tau[:i + 1], args)
        #     print(res)

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
    start = 100
    end = -1
    t_max = 100
    periodic = False

    anaIsing = AnalyticalIsing(0.1, 12, -1, 8, periodic=periodic)
    # tau, kink_density = anaIsing.tau_sweep(100)
    tau, kink_density = anaIsing.tau_sweep(t_max, res)

    # filename = f"../data/AnalyticalSolution_N{anaIsing.N}_J{anaIsing.J}_h{anaIsing.h}_dt{anaIsing.dt}_num{res}"
    filename = f"../data/AnalyticalSolution_N{anaIsing.N}_J{anaIsing.J}_h{anaIsing.h}_tmax{t_max}_num{res}_periodic{periodic}"
    with open(filename, "rb") as f:
        things_to_load = pickle.load(f)
        [tau, kink_density] = things_to_load

    popt, pcov = curve_fit(anaIsing.kink_density_theory, tau[start:end], kink_density[start:end])
    anaIsing.plot(tau, kink_density, popt[0], popt[1], popt[2])

    # U, S, V = svd(anaIsing.hamiltonian(0))
    # U2, S2, V2 = svd(anaIsing.hamiltonian(1))
    # svdexpm = U @ expm(-1j * np.diag(S)) @ np.transpose(U)
    # directexpm = expm(-1j * anaIsing.hamiltonian(0))
    # print(svdexpm-directexpm < 1e-03)
    # print(U @ np.transpose(U2))

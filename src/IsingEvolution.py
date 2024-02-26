import matplotlib.pyplot as plt
import tqdm
import numpy as np
import logging as log
from qiskit import QuantumCircuit
from qiskit.quantum_info.operators import SparsePauliOp
from qiskit_aer import Aer
from scipy.sparse.linalg import eigsh
from scipy.sparse import identity

log.basicConfig(level=log.INFO, filename="output.log")
backend = Aer.get_backend('aer_simulator_statevector')


class IsingEvol():

    def __init__(self, N: int, dt: float, h: float, J: float, gpu=False, periodic=False, inverse=False, bias=None):
        self.N = N
        self.dt = dt
        self.h = h
        self.bias_parallel = bias
        self.J = J
        self.periodic = periodic
        self.inverse = inverse
        self.progress = True
        self.linear_increase = True
        self.obs = {}

        if inverse:
            self.ground_state = self.groundState()

        self.nok = self.nok_observable()
        self.nok_square = self.nok ** 2
        self.nok_cube = self.nok ** 3

        if gpu:
            backend.set_options(device='GPU')

    def groundState(self, step: int = 0, steps: int = 1):
        log.info("Computing Ground state")

        result = []

        for i in range(self.N - 1):
            result.append(('XX', [i, i + 1], self.J))

        if self.periodic:
            result.append(('XX', [0, self.N - 1], self.J))

        for i in range(self.N):
            result.append(('Z', [i], self.h * (1 - step / steps)))

        if self.bias_parallel:
            for i in range(self.N):
                result.append(('X', [i], self.bias_parallel))

        matr = np.real(SparsePauliOp.from_sparse_list(result, num_qubits=self.N).to_matrix(sparse=True))
        eigval, eigvec = eigsh(matr, which="SA", k=1)
        log.info(f"Computed groundstate with eigenvalue {eigval}")
        return eigvec

    def nok_observable(self):
        log.info(f"Computing Observable")
        result = [('I', [1], self.N - (0 if self.periodic else 1))]

        for i in range(self.N - 1):
            result.append(('XX', [i, i + 1], -1))

        if self.periodic:
            result.append(('XX', [0, self.N - 1], -1))

        matr = np.real(SparsePauliOp.from_sparse_list(result, num_qubits=self.N).to_matrix(sparse=True)) / (2 * self.N)
        log.info(f"Observable computed")
        return matr

    def nok_variance(self, exp_val: float):
        log.info(f"Computing Variance Observable")
        result = self.nok_square - exp_val ** 2 * identity(2 ** self.N)
        log.info(f"Variance Observable computed")
        return result

    def nok_skewness(self, exp_val: float):
        log.info(f"Computing Skewness Observable")
        result = self.nok_cube - 3 * self.nok_square * exp_val + 3 * self.nok * exp_val ** 2 - exp_val ** 3 * identity(
            2 ** self.N)
        log.info(f"Skewness Observable computed")
        return result

    @staticmethod
    def observables(N, obs_idx: dict):
        log.info(f"Generating observables for Observable dictionary: {obs_idx}")
        obs = {}
        for obsstr in obs_idx:
            obs[obsstr] = {}
            for idx in obs_idx[obsstr]:
                obs_gen = [(obsstr, idx, 1)]
                obs[obsstr][tuple(idx)] = SparsePauliOp.from_sparse_list(obs_gen, num_qubits=N).to_matrix(sparse=True)
        return obs

    # Second order trotter formula
    def evolution_step(self, qc, step=1, proportion=1.0, sample_step=1, h=None):

        if self.bias_parallel:
            for idx in range(self.N):
                qc.rx(self.bias_parallel * self.dt, idx)

        for idx in range(self.N):
            qc.rz((h or self.h) * self.dt * proportion, idx)

        for idx in range(self.N - 1):
            qc.rxx(2 * self.J * self.dt, idx, idx + 1)

        if self.periodic:
            qc.rxx(2 * self.J * self.dt, self.N - 1, 0)

        for idx in range(self.N):
            qc.rz((h or self.h) * self.dt * proportion, idx)

        if self.bias_parallel:
            for idx in range(self.N):
                qc.rx(self.bias_parallel * self.dt, idx)

        if step % sample_step == 0:
            qc.save_statevector(label=str(int(step / sample_step)))

    def circuit(self, steps: int, samples=1, h=None):
        log.info(f"Building circuit for {steps} steps")
        sample_step = max(np.floor(steps / samples), 1)
        qc = QuantumCircuit(self.N)

        if self.inverse:
            qc.set_statevector(self.ground_state)
        else:
            qc.h(0)
            for idx in range(self.N - 1):
                qc.cnot(0, idx + 1)

            for idx in range(self.N):
                qc.h(idx)

        for step in tqdm.tqdm(range(1, steps + 1), disable=not self.progress):
            self.evolution_step(qc, step=step, proportion=(
                (1 - step / steps if self.inverse else step / steps) if self.linear_increase else 1),
                                sample_step=sample_step, h=h)
        return qc

    def execute(self, steps: int = 1, draw: bool = False, samples: int = 1, h: float = None):
        qc = self.circuit(steps, samples=samples, h=h)
        if draw:
            print(qc)

        job = backend.run(qc)
        res = job.result()
        log.info(f"Time taken for execution: {res.time_taken}")
        return res.data(0)

    @staticmethod
    def compute_expectationvals(obs: dict, states: dict):
        log.info(f"Computing expectation values of observables")
        expectations = {}

        for obsstr in obs:
            expectations[obsstr] = {}
            for idx in obs[obsstr]:
                expectations[obsstr][idx] = [np.real(np.conj(states[step]) @ obs[obsstr][idx] @ states[step]) for step
                                             in range(len(states))]

        return expectations

    def compute_kink_density(self, states: dict, raw: bool = False):
        log.info(f"Computing expectation, variance and skewness values for number of kinks ({len(states)} states)")
        exp_kinks = []
        var_kinks = []
        skew_kinks = []
        for step in tqdm.tqdm(range(1, len(states) + 1), disable=not self.progress):
            exp_val = np.real(np.conj(states[str(step)].data) @ self.nok @ states[str(step)].data)
            exp_kinks.append(exp_val)
            var_kinks.append(
                np.real(np.conj(states[str(step)].data) @ self.nok_variance(exp_val) @ states[str(step)].data))
            skew_kinks.append(
                np.real(np.conj(states[str(step)].data) @ self.nok_skewness(exp_val) @ states[str(step)].data))
        log.info(f"Finished computing expectation values for number of kinks ({len(states)} states)")

        if raw:
            return exp_kinks
        else:
            return np.mean(exp_kinks), np.mean(var_kinks), np.mean(skew_kinks)

    def plot(self, expectations: dict):
        log.info("Plotting observables...")

        for obsstr in expectations:
            for idx in expectations[obsstr]:
                plt.plot(expectations[obsstr][idx], label=r'$\langle \sigma^{{}}_{{{}}} \rangle $'.format(obsstr, idx))

        plt.xlabel('trotter steps')
        plt.legend()
        plt.savefig(
            f'../figs/quench_N{self.N}_lin{self.linear_increase}_J{self.J}_h{self.h}_dt{self.dt}_periodic{self.periodic}.png')

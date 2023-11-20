import matplotlib.pyplot as plt
import tqdm
import numpy as np
import logging as log
from qiskit import QuantumCircuit, execute
from qiskit.quantum_info.operators import SparsePauliOp
from qiskit import Aer
from itertools import product

log.basicConfig(level=log.INFO, filename="output.log")
backend = Aer.get_backend('aer_simulator_statevector')


class IsingEvol():

    def __init__(self, N, dt, h, J, gpu=False):
        self.N = N
        self.dt = dt
        self.h = h
        self.J = J
        self.progress = True
        self.linear_increase = True
        self.obs = {
            'Z': [],
            'XX': []
        }
        self.obs_idx = {
            'Z': [],
            'XX': []
        }

        if gpu:
            backend.set_options(device='GPU')

    def observables(self, obs_Z, obs_XX):
        operators_Z = []
        operators_XX = []

        log.info(f"Generating observables for Z: {obs_Z} and XX: {obs_XX}")
        for op in obs_Z:
            pauli_str = "I" * (op - 1) + "Z" + "I" * (self.N - op)
            operators_Z.append(SparsePauliOp(pauli_str))
        for op in obs_XX:
            pauli_str = "I" * (op[0] - 1) + "X" + "I" * (op[1] - op[0] - 1) + "X" + "I" * (self.N - op[1])
            operators_XX.append(SparsePauliOp(pauli_str))

        self.obs['Z'] = operators_Z
        self.obs['XX'] = operators_XX
        self.obs_idx['Z'] = obs_Z
        self.obs_idx['XX'] = obs_XX

    def evolution_step(self, qc, step=1, proportion=1.0):
        for idx in range(self.N):
            qc.rz(2 * self.h * self.dt * proportion, idx)

        for idx in range(self.N - 1):
            qc.rxx(2 * self.J * self.dt, idx, idx + 1)
        qc.save_statevector(label=str(step))

    def circuit(self, steps, linear_increase=True):
        log.info(f"Building circuit for {steps} steps")
        self.linear_increase = linear_increase
        qc = QuantumCircuit(self.N)
        for idx in range(self.N):
            qc.h(idx)

        for step in tqdm.tqdm(range(1, steps + 1), disable=not self.progress):
            self.evolution_step(qc, step=step, proportion=(step / steps if linear_increase else 1))
        return qc

    def execute(self, steps=1, linear_increase=True, draw=False):
        qc = self.circuit(steps, linear_increase=linear_increase)
        if draw:
            print(qc)

        job = execute(qc, backend)
        res = job.result()
        log.info(f"Time taken for execution: {res.time_taken}")
        return res.data(0)

    def compute_expectationvals(self, states):
        log.info(f"Computing expectation values of observables")
        expectations = {
            'Z': [],
            'XX': []
        }

        for step in tqdm.tqdm(range(1, len(states) + 1), disable=not self.progress):
            expectations['Z'].append([states[str(step)].expectation_value(op).real for op in self.obs['Z']])
            expectations['XX'].append([states[str(step)].expectation_value(op).real for op in self.obs['XX']])

        return expectations

    def compute_kink_density(self, states):
        spins = [0, 1]
        nok = []
        for subset in product(spins, repeat=self.N):
            arr = np.array(subset)
            nok.append(np.sum(np.abs(arr[0:-1] - arr[1:])) / self.N)

        log.info(f"Computing expectation values for number of kinks")
        exp_kinks = []
        diag_matr = np.array(nok)
        for step in tqdm.tqdm(range(1, len(states) + 1), disable=not self.progress):
            exp_kinks.append(
                np.real(np.dot(np.conj(states[str(step)].data), diag_matr * states[str(step)].data)))
        return np.mean(exp_kinks), np.var(exp_kinks)

    def plot(self, expectations):
        log.info("Plotting observables...")
        expZ = np.array(expectations['Z'])
        expXX = np.array(expectations['XX'])

        for i in range(len(self.obs_idx['Z'])):
            plt.plot(expZ[:, i], label=r'$\langle \sigma^z_{{{}}} \rangle $'.format(self.obs_idx['Z'][i]))

        for i in range(len(self.obs_idx['XX'])):
            plt.plot(expXX[:, i],
                     label=r'$\langle \sigma^X_{{{}}} \sigma^X_{{{}}} \rangle$'.format(self.obs_idx['XX'][i][0],
                                                                                       self.obs_idx['XX'][i][1]))
        plt.xlabel('trotter steps')
        plt.legend()
        plt.savefig(f'../figs/quench_N{self.N}_lin{self.linear_increase}_J{self.J}_h{self.h}_dt{self.dt}.png')

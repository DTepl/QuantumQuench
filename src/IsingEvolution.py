import matplotlib.pyplot as plt
import tqdm
import numpy as np
import logging as log
from qiskit import QuantumCircuit
from qiskit.quantum_info.operators import SparsePauliOp
from qiskit_aer import Aer
from scipy.sparse.linalg import eigsh

log.basicConfig(level=log.INFO, filename="output.log")
backend = Aer.get_backend('aer_simulator_statevector')


class IsingEvol():

    def __init__(self, N, dt, h, J, gpu=False, periodic=False, inverse=False, bias=None):
        self.N = N
        self.dt = dt
        self.h = h
        self.bias_parallel = bias
        self.J = J
        self.periodic = periodic
        self.inverse = inverse
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

        if inverse:
            self.ground_state = self.groundState()

        self.nok = self.nok_observable()

        if gpu:
            backend.set_options(device='GPU')

    def groundState(self):
        log.info("Computing Ground state")

        result = []

        for i in range(self.N - 1):
            result.append(('XX', [i, i + 1], self.J))

        if self.periodic:
            result.append(('XX', [0, self.N - 1], self.J))

        for i in range(self.N):
            result.append(('Z', [i], self.h))

        if self.bias_parallel:
            for i in range(self.N):
                result.append(('X', [i], self.bias_parallel))

        matr = SparsePauliOp.from_sparse_list(result, num_qubits=self.N).to_matrix(sparse=True)
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

        matr = SparsePauliOp.from_sparse_list(result, num_qubits=self.N).to_matrix(sparse=True) / (2 * self.N)
        log.info(f"Observable computed")
        return matr

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

    def circuit(self, steps, samples=1, h=None):
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

    def execute(self, steps=1, draw=False, samples=1, h=None):
        qc = self.circuit(steps, samples=samples, h=h)
        if draw:
            print(qc)

        job = backend.run(qc)
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

    def compute_kink_density(self, states, raw=False):
        log.info(f"Computing expectation values for number of kinks ({len(states)} states)")
        exp_kinks = []
        for step in tqdm.tqdm(range(1, len(states) + 1), disable=not self.progress):
            exp_kinks.append(
                np.real(np.conj(states[str(step)].data) @ self.nok @ states[str(step)].data))
        log.info(f"Finished computing expectation values for number of kinks ({len(states)} states)")

        if raw:
            return exp_kinks
        else:
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
        plt.savefig(
            f'../figs/quench_N{self.N}_lin{self.linear_increase}_J{self.J}_h{self.h}_dt{self.dt}_periodic{self.periodic}.png')

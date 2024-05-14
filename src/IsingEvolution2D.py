import tqdm
import numpy as np
import logging as log
from qiskit import QuantumCircuit
from qiskit.quantum_info.operators import SparsePauliOp
from qiskit_aer import Aer
from scipy.sparse.linalg import eigsh

log.basicConfig(level=log.INFO, filename="output.log")
backend = Aer.get_backend('aer_simulator_statevector')


class IsingEvol2D:

    def __init__(self, N: tuple[int, int], dt: float, h: float, J: float, gpu=False, periodic=False, inverse=False,
                 bias=None):
        self.N_x = N[0]
        self.N_y = N[1]
        self.dt = dt
        self.h = h
        self.bias_parallel = bias
        self.J = J
        self.periodic = periodic
        self.inverse = inverse
        self.progress = True
        self.linear_increase = True
        self.obs = {}

        # if inverse:
        #     self.ground_state = self.groundState()

        if gpu:
            backend.set_options(device='GPU')

    def groundState(self, step: int = 0, steps: int = 1):
        log.info("Computing Ground state")

        result = []

        # Interaction terms (nearest neighbors)
        for i in range(self.N_y):
            for j in range(self.N_x):
                # Horizontal interaction
                if j < self.N_x - 1:
                    result.append(('XX', [i * self.N_x + j, i * self.N_x + (j + 1)], self.J * step / steps))
                elif self.periodic:
                    result.append(
                        ('XX', [i * self.N_x + j, i * self.N_x],
                         self.J * step / steps))  # Wrap around for periodic boundary
                # Vertical interaction
                if i < self.N_y - 1:
                    result.append(('XX', [i * self.N_y + j, (i + 1) * self.N_y + j], self.J * step / steps))
                elif self.periodic:
                    result.append(
                        ('XX', [i * self.N_y + j, j], self.J * step / steps))  # Wrap around for periodic boundary

        for i in range(self.N_y * self.N_x):
            # External magnetic field term
            result.append(('Z', [i], self.h * (1 - step / steps)))

            if self.bias_parallel:
                # Parallel bias field term
                result.append(('X', [i], self.bias_parallel))

        matr = np.real(SparsePauliOp.from_sparse_list(result, num_qubits=self.N_x * self.N_y).to_matrix(sparse=True))
        eigval, eigvec = eigsh(matr, which="SA", k=1)
        log.info(f"Computed groundstate with eigenvalue {eigval}")
        return eigvec

    # Second order trotter formula
    def evolution_step(self, qc, step=1, proportion=1.0, sample_step=1, h=None):

        # Apply biases in the x-direction (if present)
        if self.bias_parallel:
            for idx in range(self.N_y):
                for jdx in range(self.N_x):
                    qc.rx(self.bias_parallel * self.dt * (1 - proportion), idx * self.N_x + jdx)

        # Apply local field in the z-direction
        for idx in range(self.N_y):
            for jdx in range(self.N_x):
                qc.rz((h or self.h) * self.dt * proportion, idx * self.N_x + jdx)

        # Apply nearest-neighbor interactions
        for idx in range(self.N_y):
            for jdx in range(self.N_x - 1):
                qc.rxx(2 * self.J * self.dt * (1 - proportion), idx * self.N_x + jdx, idx * self.N_x + jdx + 1)

        # Handle periodic boundary conditions (if applicable)
        if self.periodic:
            for idx in range(self.N_y - 1):
                qc.rxx(2 * self.J * self.dt * (1 - proportion), (idx + 1) * self.N_x - 1, idx * self.N_x)

        for idx in range(self.N_y - 1):
            for jdx in range(self.N_x):
                qc.rxx(2 * self.J * self.dt * (1 - proportion), idx * self.N_x + jdx, (idx + 1) * self.N_x + jdx)

        # Apply local field in the z-direction again
        for idx in range(self.N_y):
            for jdx in range(self.N_x):
                qc.rz((h or self.h) * self.dt * proportion, idx * self.N_x + jdx)

        # Apply biases in the x-direction again (if present)
        if self.bias_parallel:
            for idx in range(self.N_y):
                for jdx in range(self.N_x):
                    qc.rx(self.bias_parallel * self.dt * (1 - proportion), idx * self.N_x + jdx)

        # Optionally, save statevector for sampling
        if step % sample_step == 0:
            qc.save_statevector(label=str(int(step / sample_step)))

    def circuit(self, steps: int, samples=1, h=None):
        log.info(f"Building circuit for {steps} steps")
        sample_step = max(np.floor(steps / samples), 1)
        qc = QuantumCircuit(self.N_x * self.N_y)

        if self.inverse:
            # qc.set_statevector(self.ground_state)
            if self.h > 0:
                for idx in range(self.N_x * self.N_y - 1):
                    qc.x(idx)
        else:
            qc.h(0)
            for idx in range(self.N_x * self.N_y - 1):
                qc.cnot(0, idx + 1)

            for idx in range(self.N_x * self.N_y):
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

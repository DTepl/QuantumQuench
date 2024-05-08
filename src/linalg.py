import numpy as np
import logging as log
from qiskit.quantum_info.operators import SparsePauliOp
from qiskit.quantum_info import partial_trace, Statevector


def observables(N, obs_idx: dict):
    log.info(f"Generating observables for Observable dictionary: {obs_idx}")
    obs = {}
    for obsstr in obs_idx:
        obs[obsstr] = {}
        for idx in obs_idx[obsstr]:
            obs_gen = [(obsstr, idx, 1)]
            obs[obsstr][tuple(idx)] = SparsePauliOp.from_sparse_list(obs_gen, num_qubits=N).to_matrix(sparse=True)
    return obs


def compute_expectationvals(obs: dict, states: dict):
    log.info(f"Computing expectation values of observables")
    expectations = {}

    for obsstr in obs:
        expectations[obsstr] = {}
        for idx in obs[obsstr]:
            expectations[obsstr][idx] = np.array(
                [np.real(np.conj(states[step]) @ obs[obsstr][idx] @ states[step]) for step in range(len(states))])

    return expectations


def compute_partial_trace(N: int, data: list[list], keepsub: list):
    tr_sub_sys = list(range(N))
    for sys in keepsub:
        tr_sub_sys.remove(sys)

    res = []
    for state in data:
        vec = Statevector(state)
        res.append(partial_trace(vec, tr_sub_sys).data)

    return res

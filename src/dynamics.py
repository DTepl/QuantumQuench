import cirq
import numpy as np
import tqdm
import pennylane as qml

from src.IsingEvolution import IsingEvol

N = 16
J = 0.8
h = 1
dt = 0.1  # time step
obs_Z = [2, 15]
obs_XX = [[1, 13], [5, 14]]

ising_model_evolution = IsingEvol(N, dt, h, J)
print(qml.draw(ising_model_evolution.circuit())())

results = []
for n in tqdm.tqdm(range(1, 200)):
    results.append(ising_model_evolution.circuit()(obs_Z=obs_Z, obs_XX=obs_XX, steps=n))
results = np.array(results)

print(results.shape)
ising_model_evolution.plot(results, obs_Z, obs_XX)

from src.IsingEvolution import IsingEvol

N = 20
steps = 200
J = -0.5
h = -0.25
dt = 0.1  # time step
obs_Z = [5, 6]
obs_XX = [[4, 14], [5, 6]]

ising_model_evolution = IsingEvol(N, dt, h, J)

ising_model_evolution.observables(obs_Z, obs_XX)
# ising_model_evolution.execute(draw=True, steps=200)
ising_model_evolution.execute(draw=True, linear_increase=False, steps=200)
ising_model_evolution.compute_expectationvals()
ising_model_evolution.plot()

import pennylane as qml
import matplotlib.pyplot as plt


class IsingEvol():

    def __init__(self, N, dt, h, J):
        self.N = N
        self.dt = dt
        self.h = h
        self.J = J
        self.dev = qml.device("lightning.qubit", wires=range(N))

    def evolution_step(self):
        for idx in range(self.N):
            qml.RZ(2 * self.h * self.dt, wires=idx)

        for idx in range(self.N - 1):
            qml.IsingXX(2 * self.J * self.dt, wires=[idx, idx + 1])

    def quench_process(self, obs_Z=[0], obs_XX=[], steps=1):
        for idx in range(self.N):
            qml.Hadamard(idx)

        for _ in range(steps):
            self.evolution_step()
        return [qml.expval(qml.PauliZ(i)) for i in obs_Z] + [
            qml.expval(qml.PauliX(i[0]) @ qml.PauliX(i[1])) for i in obs_XX]

    def circuit(self):
        return qml.QNode(self.quench_process, self.dev)

    def plot(self, data, obs_Z, obs_XX):
        for i in range(len(obs_Z)):
            plt.plot(data[:, i], label=r'$\langle \sigma^z_{{}} \rangle $'.format(obs_Z[i]))
        for i in range(len(obs_XX)):
            plt.plot(data[:, len(obs_Z) + i],
                     label=r'$\langle \sigma^X_{{{}}} \sigma^X_{{{}}} \rangle$'.format(obs_XX[i][0], obs_XX[i][1]))
        plt.xlabel('trotter steps')
        plt.legend()
        plt.savefig(f'../figs/quench_N{self.N}_J{self.J}_h{self.h}_dt{self.dt}.png')

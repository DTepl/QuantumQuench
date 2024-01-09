# Quantum Quench of 1D-Ising Chain

## How to install

### Create and activate the environment

1. ```python3 -m venv <env-name>```
2. ```source <env_name>/bin/activate```

### Clone and move to Project folder

3. ```git clone https://github.com/DTepl/QuantumQuench.git```
4. ```cd QuantumQuench```

### Install required packages

5. ```pip install ./```

---

### Install GPU support (Not implemented)

6. ```pip install qiskit-aer-gpu```
7. If error ```https://github.com/Qiskit/qiskit-aer/issues/1874``` still not fixed, install
   via ```pip install -i https://test.pypi.org/simple/ qiskit-aer-gpu==0.13.0```

---

## Run program

You can either run simulations on a quantum computer or compute the analytical predictions

### Simulations on a quantum computer

Run the following command to do kink density estimation
`python src/dynamics.py 16 100 -s=1 -Jv -1 -hv 2 -b 0 -dt 0.1 -m 0 -s 1 -p 0 -inv 1`
For further informations on the parameters run 
`python src/dynamics.py -h`

### Analytical computations
Run the following command to do kink density estimation
`python src/analytical.py`
Unfortunately this module is not that advanced developed and the parameters need to be set in the code
The most important are:
* res: Resolution of tau's to sample over for the plot
* t_max: maximal tau (the lowest will be t_max/res)
* N: Number of spins
* J: Coupling strength
* h: external magnetic field

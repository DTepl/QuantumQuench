import numpy as np
from scipy.optimize import curve_fit


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def derivation(data):
    return np.gradient(data)


def kink_density_theory(tau, a, e, g):
    return a * tau ** e * np.exp(-tau * g)


def entropy_3s_theory(tau, a, b, c, d):
    return -a * np.tanh(-(c * tau) ** b) + d


def fit_kinks(tau, kinks):
    return curve_fit(kink_density_theory, tau, kinks, maxfev=5000)


def fit_entropy(tau, entropy):
    return curve_fit(entropy_3s_theory, tau, entropy, maxfev=5000)

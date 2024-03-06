import numpy as np
from scipy.optimize import curve_fit


def kink_density_theory(tau, a, e, g):
    return a * tau ** e * np.exp(-tau * g)


def fit_kinks(tau, kinks):
    return curve_fit(kink_density_theory, tau, kinks, maxfev=5000)

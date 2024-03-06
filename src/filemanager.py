import pickle


def save_file(filename, data):
    with open("../data/" + filename, "wb") as f:
        pickle.dump(data, f)


def load_file(filename):
    with open(filename, "rb") as f:
        things_to_load = pickle.load(f)

    [tau, kdens_mean, kdens_var, *kdens_skewness] = things_to_load
    return tau, kdens_mean, kdens_var, (kdens_skewness or [None])[0]

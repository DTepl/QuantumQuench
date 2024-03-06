import numpy as np


def pair_to_distance(N: int, pair_dict: dict, periodic=False):
    distance_dict = {}
    for idx in pair_dict:
        distance = np.abs(idx[1] - idx[0])

        #  Indeces are mirrored at the 'largest' distance possible due to periodicity.
        #  e.g. N=10 largest distance possible is 5 as if 6 it is already 4 in the 'other' direction
        if periodic and distance > int(N / 2):
            distance = N - distance

        if not tuple([distance]) in distance_dict:
            distance_dict[tuple([distance])] = []

        distance_dict[tuple([distance])].append(pair_dict[idx].copy())

    return distance_dict


def correlator_processing(N, correlators, periodic=False):
    result = {}
    for quench_time in correlators:
        result[quench_time] = {}
        for obsstr in correlators[quench_time]:
            result[quench_time][obsstr] = correlators[quench_time][obsstr].copy()

            if len(obsstr) == 2:
                result[quench_time][obsstr] = pair_to_distance(N, correlators[quench_time][obsstr], periodic=periodic)

    return result


def entropies_processing(N, entropies, periodic=False):
    result = {}
    for quench_time in entropies:
        result[quench_time] = pair_to_distance(N, entropies[quench_time], periodic=periodic)

    return result

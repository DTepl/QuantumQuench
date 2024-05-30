import numpy as np


def pair_to_distance(N: tuple[int, int], pair_dict: dict, periodic=False):
    distance_dict = {}
    if isinstance(N, int):
        N = (N, 0)

    for idx in pair_dict:
        y1, x1 = divmod(idx[0], N[0])
        y2, x2 = divmod(idx[1], N[0])

        dx = np.abs(x2 - x1)
        dy = np.abs(y2 - y1)
        distance = np.sqrt(dx ** 2 + dy ** 2)

        #  Indeces are mirrored at the 'largest' distance possible due to periodicity.
        #  e.g. N=10 largest distance possible is 5 as if 6 it is already 4 in the 'other' direction
        if periodic:
            dx_wrap = min(dx, N[0] - dx)
            dy_wrap = min(dy, (N[1] if N[1] else 0) - dy)
            distance = np.sqrt(dx_wrap ** 2 + dy_wrap ** 2)

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

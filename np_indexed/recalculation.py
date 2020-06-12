from functools import partial

import numpy as np
import numpy_indexed as npi
from scipy.stats import gmean


def recalculate_data(data, clusters):
    calc_average_change = partial(np.apply_along_axis, func1d=gmean, axis=0)
    clustered = np.empty((clusters.max(), data.shape[1]))
    groups = npi.group_by(clusters).split(data)
    for i, clust in enumerate(groups):
        clustered[i] = calc_average_change(arr=clust)
    return clustered


def recalculate_returns(returns, clusters):
    calc_average_change = partial(np.apply_along_axis, func1d=gmean, axis=0)
    clustered = np.empty((clusters.max(),))
    groups = npi.group_by(clusters).split(returns)
    for i, clust in enumerate(groups):
        clustered[i] = calc_average_change(arr=clust)
    return clustered